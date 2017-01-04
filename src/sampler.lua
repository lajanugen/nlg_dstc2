local sampler = torch.class('sampler')

function sampler:__init(params,sampling_params,fwd_mdl,bwd_mdl,data)

	self.params = params
	self.fwd_mdl = fwd_mdl
	self.bwd_mdl = bwd_mdl
	self.sampling_params = sampling_params
	self.data = data

	self:sample_setup() --Sets up self.model
end


function sampler:sample_setup()
	--local core_network = create_network()

	--local core_network = self:sample_network(params)
	local core_network = self:sample_network_orig(params)
	local paramx, paramdx = core_network:getParameters()

	--model = torch.load('model_20')
	--if mdl then 
	--model = self.fwd_mdl
	--local param, gradParams = model.rnns[1]:parameters()
	----local param = self.fwd_mdl
	--param_copy = {}
	--for k,v in ipairs(param) do
	--	local t = torch.Tensor(v:size())
	--	t:copy(v)
	--	table.insert(param_copy,t)
	--end

	--print(paramx:size())
	--print(flatten(param_copy):size())
	--paramx:copy(flatten(param_copy))
	
	paramx:copy(self.fwd_mdl)
	
	--end

	local model = {}
	model.core_network = core_network
	model.rnns = g_cloneManyTimes(model.core_network, self.params.seq_length)
	model.norm_dw = 0
	model.err = transfer_data(torch.zeros(params.seq_length))

	self.model = model
end

function sampler:sample_network_orig()

	local params = self.params

	local x                = nn.Identity()()
	local y                = nn.Identity()()
	local prev_s           = nn.Identity()()
	local prev_d           = nn.Identity()()
	-- i[0] = embedding vector of x
	-- i is used to store the output h of the lstm after each time step
	--local i                = {[0] = LookupTable(params.vocab_size,
	--                                                  params.input_size)(x)}
	local i
	if params.pretrained == 1 then
		local embedding = GloVeEmbeddingFixed(self.data.vocab_map, 300, '')--, 'restaurant_vectors')
		i = {[0] = embedding(x)}
	elseif params.pretrained == 2 then
		local embedding = GloVeEmbedding(self.data.vocab_map, 300, '')--, 'restaurant_vectors')
		i = {[0] = embedding(x)}
	else
		--i                = {[0] = LookupTable(params.vocab_size,params.input_size)(x)}
		i                = {[0] = nn.Linear(params.vocab_size,params.rnn_size)(x)}
	end

	local next_s           = {}
	local next_d	         = {}
	local split         = {prev_s:split(2 * params.layers)}

	--local read_gate, next_d = read_gate_upd(i[0], split, params, prev_d)
	local next_d = read_gate_upd(i[0], split, params, prev_d)

	for layer_idx = 1, params.layers do
		local prev_c         = split[2 * layer_idx - 1]
		local prev_h         = split[2 * layer_idx]
		local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
		local next_c, next_h
		if layer_idx == 1 then
			next_c, next_h		 = lstm_orig_2inp(dropped, prev_c, prev_h, params, next_d)
			--local next_c, next_h = lstm(dropped, prev_c, prev_h)
		else
			next_c, next_h		 = lstm_orig_3inp(dropped, i[0], prev_c, prev_h, params, next_d)
		end
		table.insert(next_s, next_c)
		table.insert(next_s, next_h)
		i[layer_idx] = next_h
	end

    local h2y, dropped, softmax_in
    h2y            = nn.Linear(params.rnn_size, params.vocab_size)
    dropped        = nn.Dropout(params.dropout)(i[1])
    softmax_in	 = h2y(dropped)
    local j
    for j = 2,params.layers do
      h2y            = nn.Linear(params.rnn_size, params.vocab_size)
    	dropped        = nn.Dropout(params.dropout)(i[j])
    	softmax_in	   = nn.CAddTable()({h2y(dropped), softmax_in})
    end
	
	local logpred          = nn.LogSoftMax()(softmax_in)
	--local cross_ent        = nn.ClassNLLCriterion()({logpred, y})

	local dist_d		   = nn.PairwiseDistance(2)({next_d,prev_d})
	local dist_exp		   = nn.Exp()(dist_d)
	local dist_pow		   = nn.Power(math.log(params.xi))(dist_exp)
	local dist_pen		   = nn.MulConstant(params.eta)(dist_pow)
	local d_err			   = nn.MulConstant(1.0/params.batch_size)(nn.Sum()(dist_pen))
	--local d_err			   = nn.Sum()(dist_pen)
	--local err			   = nn.CAddTable()({cross_ent, nn.Sum()(dist_pen)})
	local err = nn.Sum()(y)

    local pred			   = nn.SoftMax()(softmax_in)
	local nlogpred         = nn.MulConstant(-1)(logpred)

	local module           = nn.gModule({x, y, prev_s, prev_d},
	{err, nn.Identity()(next_s), nn.Identity()(next_d), pred, nlogpred, d_err})
	module:getParameters():uniform(-params.init_weight, params.init_weight)
	return transfer_data(module)

end

function sampler:sample(d,dsv)

	local model = self.model
	local params = self.params
	local stoch = self.sampling_params.stochastic_sampling

	local gold_pos = 0
	local generated_sentences = {}
	local generated_sentences_d = {}
	local generated_sentences_ids = {}
	local generated_sentences_delex = {}
	local sentence_words_ids = {}
	local sent
	local errors = {}
	local batch
	--local d,prev_d
	--local len = data:get_uniq_batch_count(train_test_valid,1)
	--for sent = 1,feats:size(1)/params.cond_len do
		--for batch = 1,feats:size(2) do
	--for sent = 1,len do
			--local sequences = data:get_next_uniq_batch(train_test_valid,1)
			--local ids = sequences[4]
			local err_sum = 0
			local sentence = ''
			local sentence_words = {}

			local s = {}
			for t = 1, 2 * params.layers do
				s[t] = transfer_data(torch.zeros(1, params.rnn_size))
			end

			local i = 1
			local x = self.data.START_SYMBOL
			--d = feats:sub((sent-1)*params.cond_len+1,sent*params.cond_len,batch,batch)
			--d = sequences[2]
			--local sent_id = state.ids:sub(sent,sent,batch,batch)
			d = d:transpose(1,2)
			local cond_d = d
			local within_sentence = true
			local err, pred
			while within_sentence and (i < params.seq_length)  do
				--io.write(vocab_imap[x[1]])
				local word = torch.zeros(1,params.vocab_size)
				local word0 = torch.zeros(1,params.vocab_size)
				word0[1][2] = 1
				word[1][x] = 1
				word = transfer_data(word)
				word0 = transfer_data(word0)

				prev_d = d
				_, s, d, predprob, nlogpredprob, d_err = unpack(self.model.rnns[i]:forward({word, word, s, d}))
				--err_sum = err_sum + err[1]
				cond_d = torch.cat(cond_d:float(),d:float(),1)
				--d = d:transpose(1,2)

				i = i + 1

				if stoch == 1 then
					--torch.manualSeed(5555)
					--For fwd model, we should get the logprob of the particular next word that was chosen and add it to our total error
					--For bwd model, we should simply get the error from the gModule since we are giving it both the input and the output and Crossentropy gives us the corresponding logprob
					--x = torch.multinomial(predprob, 1, true)
					x = torch.multinomial(predprob, 1)
					local chosen_word = x[1][1]
					err = nlogpredprob:sub(1,1,chosen_word,chosen_word) 
					--local d_err = torch.sum(torch.pow(params.xi,torch.abs(d - prev_d)))*params.eta
					--print(x)
				else
					err,x = torch.min(nlogpredprob,2)
				end

				err_sum = err_sum + err[1][1] + d_err[1]
				x = x[1][1]

				if x == self.data.STOP_SYMBOL then
					within_sentence = false
				else
					sentence = sentence .. self.data.ivocab_map[x] .. ' '
					table.insert(sentence_words,x)
				end

			end
			table.insert(generated_sentences_delex,sentence)
			local sent_lex = self:lexicalize(sentence,dsv)
			table.insert(generated_sentences,sent_lex)
			--table.insert(generated_sentences_ids,ids[1])
			table.insert(generated_sentences_d,cond_d)
			table.insert(sentence_words_ids,sentence_words)
			-- Add norm of last d
			local d_sum = torch.sum(torch.abs(d))
			table.insert(errors,err_sum + d_sum)
	--end

	--return generated_sentences, generated_sentences_d, sentence_words_ids, errors
	--return generated_sentences, sentence_words_ids, errors, generated_sentences_ids, generated_sentences_delex
	return generated_sentences[1]
end

function sampler:sample_fp(train_test_valid)

	local model = self.model
	local params = self.params
	local stoch = self.sampling_params.stochastic_sampling

	local gold_pos = 0
	local generated_sentences = {}
	local generated_sentences_d = {}
	local generated_sentences_ids = {}
	local generated_sentences_delex = {}
	local sentence_words_ids = {}
	local sent
	local errors = {}
	local batch
	local d,prev_d
	local len = data:get_uniq_batch_count(train_test_valid,1)
	--for sent = 1,feats:size(1)/params.cond_len do
		--for batch = 1,feats:size(2) do
	for sent = 1,len do
			local sequences = data:get_next_uniq_batch(train_test_valid,1)
			local ids = sequences[4]
			local err_sum = 0
			local sentence = ''
			local sentence_words = {}

			local s = {}
			for t = 1, 2 * params.layers do
				s[t] = transfer_data(torch.zeros(1, params.rnn_size))
			end

			local i = 1
			local x = self.data.START_SYMBOL
			--d = feats:sub((sent-1)*params.cond_len+1,sent*params.cond_len,batch,batch)
			d = sequences[2]
			--local sent_id = state.ids:sub(sent,sent,batch,batch)
			d = d:transpose(1,2)
			local cond_d = d
			local within_sentence = true
			local err, pred
			while within_sentence and (i < params.seq_length)  do
				--io.write(vocab_imap[x[1]])
				local word = torch.zeros(1,params.vocab_size)
				local word0 = torch.zeros(1,params.vocab_size)
				word0[1][2] = 1
				word[1][x] = 1
				word = transfer_data(word)
				word0 = transfer_data(word0)

				prev_d = d
				_, s, d, predprob, nlogpredprob, d_err = unpack(self.model.rnns[i]:forward({word, word, s, d}))
				--err_sum = err_sum + err[1]
				cond_d = torch.cat(cond_d:float(),d:float(),1)
				--d = d:transpose(1,2)

				i = i + 1

				if stoch == 1 then
					--torch.manualSeed(5555)
					--For fwd model, we should get the logprob of the particular next word that was chosen and add it to our total error
					--For bwd model, we should simply get the error from the gModule since we are giving it both the input and the output and Crossentropy gives us the corresponding logprob
					--x = torch.multinomial(predprob, 1, true)
					x = torch.multinomial(predprob, 1)
					x = x[1][1]
					local chosen_word = x
					if chosen_word < 1 then 
						chosen_word = 1 
						x = 1
					end
					err = nlogpredprob[1][chosen_word]
					--local d_err = torch.sum(torch.pow(params.xi,torch.abs(d - prev_d)))*params.eta
				else
					err, x = torch.min(nlogpredprob,2)
					err = err[1][1]
					x = x[1][1]
				end

				err_sum = err_sum + err + d_err[1]

				if x == self.data.STOP_SYMBOL then
					within_sentence = false
				else
					sentence = sentence .. self.data.ivocab_map[x] .. ' '
					table.insert(sentence_words,x)
				end

			end
			table.insert(generated_sentences_delex,sentence)
			local sent_lex = self:lexicalize(sentence,self.data.dacts_sv[ids[1]])
			table.insert(generated_sentences,sent_lex)
			table.insert(generated_sentences_ids,ids[1])
			table.insert(generated_sentences_d,cond_d)
			table.insert(sentence_words_ids,sentence_words)
			-- Add norm of last d
			local d_sum = torch.sum(torch.abs(d))
			table.insert(errors,err_sum + d_sum)
	end

	--return generated_sentences, generated_sentences_d, sentence_words_ids, errors
	return generated_sentences, sentence_words_ids, errors, generated_sentences_ids, generated_sentences_delex
end

function sampler:bwd_error(feats,sent)
	local mdl = self.bwd_mdl	

	within_sentence = true
	--for i = 1, params.seq_length do
	local len = sent:size(1)
	local s = {}
	for d = 1, 2 * params.layers do
		s[d] = transfer_data(torch.zeros(1, params.rnn_size))
	end

	local err_sum  = 0
	local d = feats:transpose(1,2)
	for i = 1,len-1 do
		local x = transfer_data(sent:sub(i,i))
		local y = transfer_data(sent:sub(i+1,i+1))
		err, s, d = unpack(mdl.rnns[i]:forward({x, y, s, d}))
		err_sum = err_sum + err[1]
	end
	err_sum = err_sum + torch.sum(torch.abs(d)) --norm(d) at end of sequence
	return err_sum
end

--function sampler:sample(feats,d_sv)
--
--	local params = self.params
--	local fwd_mdl = self.fwd_mdl
--	local bwd_mdl = self.bwd_mdl
--	local n = self.sampling_params.overgen
--	local m = self.sampling_params.rank
--
--	local fb_errors = {}
--	local rank_sent = {}
--	local rank_sent_lex = {}
--	for i = 1,n do
--
--		local samples,_,samples_words_ids,fwd_errors = self:sample_fp(feats)
--
--		local sample = samples_words_ids[1]
--
--		local fwd_err = fwd_errors[1]
--		local sent_lex, slot_err = self:lexicalize(samples[1],d_sv)
--
--		local total_err
--
--		if bwd_mdl then
--			-- Building the reverse sentence
--			sample_rev = {self.START_SYMBOL}
--			for ct = 1,#sample do
--				table.insert(sample_rev,sample[#sample + 1 - ct])
--			end
--			table.insert(sample_rev,self.STOP_SYMBOL)
--
--			local sample_tensor = torch.Tensor(sample_rev)
--			sample_tensor:reshape(#sample+2,1)
--
--			bwd_err = self:bwd_error(feats,sample_tensor)
--			total_err = fwd_err + bwd_err + slot_err
--		else
--			total_err = fwd_err + slot_err
--		end
--
--		rank_sent[total_err] = samples[1]
--		rank_sent_lex[total_err] = sent_lex
--		table.insert(fb_errors,total_err)
--	end
--
--	table.sort(fb_errors)
--	sorted_samples = {}
--	for j = 1,m do
--		local sent = rank_sent_lex[fb_errors[j]]
--		table.insert(sorted_samples,sent)
--	end
--
--	return sorted_samples, fb_errors
--end

function sampler:sample_old(feats,d_sv)

	local params = self.params
	local fwd_mdl = self.fwd_mdl
	local bwd_mdl = self.bwd_mdl
	local n = self.sampling_params.overgen
	local m = self.sampling_params.rank

	local samples,_,samples_words_ids,fwd_errors = self:sample_fp(feats)
	io.output(io.stdout)
	print(samples)

	local fb_errors = {}
	local rank_sent = {}
	local rank_sent_lex = {}
	for i = 1,n do
		local sample = samples_words_ids[i]

		local fwd_err = fwd_errors[i]
		local sent_lex, slot_err = self:lexicalize(samples[i],d_sv)

		local total_err

		if bwd_mdl then
			-- Building the reverse sentence
			sample_rev = {self.START_SYMBOL}
			for ct = 1,#sample do
				table.insert(sample_rev,sample[#sample + 1 - ct])
			end
			table.insert(sample_rev,self.STOP_SYMBOL)

			local sample_tensor = torch.Tensor(sample_rev)
			sample_tensor:reshape(#sample+2,1)

			bwd_err = self:bwd_error(feats,sample_tensor)
			total_err = fwd_err + bwd_err + slot_err
		else
			total_err = fwd_err + slot_err
		end

		rank_sent[total_err] = samples[i]
		rank_sent_lex[total_err] = sent_lex
		table.insert(fb_errors,total_err)
	end

	table.sort(fb_errors)
	sorted_samples = {}
	for j = 1,m do
		local sent = rank_sent_lex[fb_errors[j]]
		table.insert(sorted_samples,sent)
	end

	return sorted_samples, fb_errors
end

function sampler:lexicalize(sent_lex,dacts_sv)
	local num_redundant_slots = 0
	local i = string.find(sent_lex,'value')
	local slot
	local dacts_sv_count = {}
	--print(dacts_sv)
	for s,v in pairs(dacts_sv) do
		dacts_sv_count[s] = 1
	end
	while i do
		local j = string.find(sent_lex,' ',i)
		local value = string.sub(sent_lex,i,j-1)
		slot = self.data.islot_index[value]
		if slot == nil then
			slot = 'UNKNOWN'
			--num_redundant_slots = num_redundant_slots + 1
		end
		if dacts_sv[slot] == nil then
			sent_lex = string.sub(sent_lex,1,i-1) .. '<' .. slot .. '> ' .. string.sub(sent_lex,j,-1)
			num_redundant_slots = num_redundant_slots + 1
		else
			local val
			--print(dacts_sv_count)
			--print(slot)
			if dacts_sv_count[slot] > #dacts_sv[slot] then
				val = 'UNKNOWN'
			else
				val = dacts_sv[slot][dacts_sv_count[slot]]
				dacts_sv_count[slot] = dacts_sv_count[slot] + 1
			end
			if val == 'dontcare' then
				print('###########################')
			end
			sent_lex = string.sub(sent_lex,1,i-1) .. val .. string.sub(sent_lex,j,-1)
		end
		i = string.find(sent_lex,'value')
	end
	return sent_lex, params.redundant_slot_penalty*num_redundant_slots
end


function sampler:reverse(sent_bwd)
	local sent_rev = string.split(sent_bwd," ")
	local sent_fwd = ''
	for i = 1,#sent_rev do
		sent_fwd = sent_fwd .. sent_rev[#sent_rev + 1 - i]
		--if i ~= #sent_rev then
		sent_fwd = sent_fwd .. ' ' 
		--end
	end

	return sent_fwd
end

function sampler:gen_lex(generated_sentences,state,epoch)

	local params = self.params

	local sent_lex_tokens = {}
	local results_file = io.open(params.save_path .. state.name .. "/gen/" .. tostring(epoch),"w")
	io.output(results_file)
	local id
	local s,sent
	local count = 1
	local size = state.data:size(2)
	for s,sent in ipairs(generated_sentences) do
		--io.output(io.stdout)
		--print(state.ids[math.ceil(s/params.batch_size)])
		--id = state.ids[math.ceil(s/size)]:sub(1,1,count,count)[1][1]
		id = state.ids[math.ceil(s/size)][count]
		count = count + 1
		if count > size then
			count = 1
		end

		if params.bwd then
			sent = self:reverse(sent)
		end

		local sent_lex = sent

		local tokens = string.split(sent," ")
		for i = 1,#tokens do
			if self.data.islot_index[tokens[i]] ~= nil then
				local slot = self.data.islot_index[tokens[i]]
				if slot == nil then 
					tokens[i] = 'UNKNOWN'
				else
					tokens[i] = self.data.dacts_sv[id][slot]
				end
			end
		end
		table.insert(sent_lex_tokens,tokens)

		sent_lex = self:lexicalize(sent,self.data.dacts_sv[id])

		io.write(self.data.dact[id])			io.write('*')
		io.write(sent_lex)						io.write('*')
		io.write(self.data.raw_utt[id])			io.write('*')
		io.write(sent)							io.write('*')
		io.write(self.data.raw_utt_delex[id])	io.write('*')
		io.write('\n')
	end
	io.close(results_file)
	io.output(io.stdout)
	return sent_lex_tokens
end

function sampler:writeToFile(train_test_valid, sentences, sent_delex, ids, epoch)
	local results_file = io.open(params.save_path .. train_test_valid .. "/gen/" .. tostring(epoch),"w")
	for i = 1,#ids do
		local id = ids[i]
		results_file:write(self.data.dact[id])					results_file:write('*')
		results_file:write(sentences[i])						results_file:write('*')
		results_file:write(self.data.raw_utt[id])				results_file:write('*')
		results_file:write(sent_delex[i])						results_file:write('*')
		results_file:write(self.data.raw_utt_delex[id])			results_file:write('*')
		results_file:write('\n')
	end
	results_file:close()
end


function sampler:plot_rdgate(sen_id, sent_lex_tokens,generated_sentences,generated_sentences_d)

	sen_id = 128
	cond_ds = generated_sentences_d[sen_id]
	print(cond_ds:sub(cond_ds:size(1),cond_ds:size(1),1,22):sum())
	local d = generated_sentences_d[sen_id]
	local s = generated_sentences[sen_id]
	local plot = {}
	local Ndacts = #dact_number_
	for i=1,d:size(2) do
		if d:sub(1,1,i,i)[1][1] > 0.1 then
			if i <= Ndacts then
				table.insert(plot,{dact_number_[i], d:sub(1,d:size(1),i,i)})
			else
				table.insert(plot,{slot_number_[i-Ndacts], d:sub(1,d:size(1),i,i)})
			end
		end
	end

	-- plotting job --
	local w = sent_lex_tokens[sen_id]
	--print(w)
	tics = 'set xtics ("START" 1, '
	for i=1,#w do
		tics = tics .. '"' .. w[i] .. '" ' .. tostring(i+1) .. ', '
	end
	tics = tics .. '"STOP"' .. tostring(#w+2) .. ')'
	gnuplot.raw('set xtics rotate by 45 offset -0.8,-1.8')
	gnuplot.raw('set bmargin 3')
	gnuplot.raw(tics)
	--gnuplot.raw('set key right top')
	gnuplot.raw('set key 500,500')
	--gnuplot.title(dact[sentence_ids[sen_id]:sub(1,1,1,1)[1][1]])
	gnuplot.title(dact[sentence_ids[sen_id][1]])
	gnuplot.raw('set terminal pngcairo size 1300,600')
	gnuplot.raw('set output "plot1.png"')
	gnuplot.plot(plot)
end

