
local ok,cunn = pcall(require, 'fbcunn')
if not ok then
    ok,cunn = pcall(require,'cunn')
    if ok then
        --print("warning: fbcunn not found. Falling back to cunn") 
        LookupTable = nn.LookupTable
    else
        print("Could not find cunn or fbcunn. Either is required")
        os.exit()
    end
else
    deviceParams = cutorch.getDeviceProperties(1)
    cudaComputeCapability = deviceParams.major + deviceParams.minor/10
    LookupTable = nn.LookupTable
end
require('nngraph')
require('utils.base')
require('gnuplot')
require('models.GloVeEmbedding')
require('utils.flatten')
require('models.lstms')
require('optim')
require('grad_check')
require('networks')
package.path = package.path .. ';../../?.lua'
require('utils')

opt = {	optimizer='sgd',
		learning_rate=0.001,
		decay_rate=0.95,
		momentum=0.5
	}
		
params = {		run_flow=true,
				generate_test=false,
				--grad_check_eps=0.00015--1e-3
	
				batch_size=50,
				input_size=100,
                rnn_size=100,
                layers=1,
				alphas={1,1},
                seq_length=40,

				obj_exp_term=true,
				obj_dnorm=true,

				care_values=true,

				xi=100,
				eta=0.0001,

                dropout=0,
                decay=1.05,
                init_weight=0.1,
                lr=0.1,
				lr_batch=0.1,
				lr_sgd=0.001,
				lr_decay = false,
                max_grad_norm=5,

                max_max_epoch=100,
                max_epoch=10,

				pretrained=0,
				redundant_slot_penalty=10,
				bwd=false,
				bwd_rerank=false,

				log_dir=true,
				stats_freq=1,
				log_err=true,
				log_bleu=false,
				log_err_freq=5,
				log_bleu_freq=1,
				err_log={true,true,false},
				bleu_log={true,true,false},
				run_log={true,true,false},
				make_plots=false,
				model_checkpoint=true,
				model_checkpoint_freq=10,
				run_valid=true,
				compute_bleu=true,
				train_bleus_log=false,
				valid_bleus_log=false,
				data_debug_print=false,

				old_data=false,
				old_sampler=false,

				data_unique_instances=true,
				no_splits=false,
				shuffle=true,
				data_split_ratio={0.6,0.2,0.2},

				custom_optimizer=false,
				pre_init_mdl=false,
				pre_init_mdl_path='../results/19/models/fwd_model60',
				--pre_init_mdl_path='../results/results_test/44/models/fwd_model',

				grad_checking=false,
				--grad_check_eps=1e-3
				grad_check_eps=0.005,
                vocab_size=482,
				cond_len=22,
				bleu_ref=true,
				err_display=false
			}
sampling_params = {	lexicalize=true,
					rerank=true,
					overgen = 10,
					rank = 1,
					stochastic_sampling = 1
				}

TRAIN_VAL_TEST = {'train','val','test'}

-- Folder management --
local description_file = '../results/descriptions'
local descriptions_read = io.open(description_file, 'r')
local descriptions_write = io.open(description_file, 'a')
local last_line
for line in descriptions_read:lines() do if line then last_line = line end end
line = string.split(last_line," ")
g_init_gpu(arg)
if #arg > 0 then
	if arg[1] == '0' then
		results_number = line[1]
	else
		results_number = tostring(tonumber(line[1]) + 1)
		descriptions_write:write(results_number .. ' ' .. arg[1] .. '\n')
	end
	arg[1] = nil
else
	error("Enter description")
end
save_path = '../results/' .. results_number .. '/'
descriptions_read:close()
descriptions_write:close()
print('Directory done')
-------------------------

if params.old_sampler then
	require('old_sampler')
else
	require('sampler')
end

if params.old_data then
	require('data_old_wrapper')
else
	require('data')
	--require('data_Jan9')
end

function transfer_data(x)
  return x:cuda()
end


function setup()
  print("Creating a RNN LSTM network.")
  --local core_network = create_network()
  --local core_network = create_network_cond()
  
  local model = {}
  local core_network
  if params.pre_init_mdl then
      model = torch.load(params.pre_init_mdl_path)
      core_network = model.core_network
  else
  	  core_network = create_network_orig3()
      model.core_network = core_network
  	  model.rnns = g_cloneManyTimes(core_network, params.seq_length)
  	  model.norm_dw = 0
  	  model.err = transfer_data(torch.zeros(params.seq_length))
  end

  --local core_network = create_network_orig()
  paramx, paramdx = core_network:getParameters()
  model.paramx = paramx
  model.paramdx = paramdx
  model.s = {} -- stores the c,h states of the LSTMs 
  model.d = {} -- stores the c,h states of the LSTMs 
  model.ds = {} -- stores derivatives
  model.dd = {} -- stores derivatives
  model.start_s = {}
  model.start_d = {}
  for j = 0, params.seq_length do
    model.s[j] = {}
    model.d[j] = transfer_data(torch.zeros(params.batch_size, params.cond_len))
    for d = 1, 2 * params.layers do
      model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
  end
  model.start_d = transfer_data(torch.zeros(params.batch_size, params.cond_len))
  model.dd = transfer_data(torch.zeros(params.batch_size, params.cond_len))
  for d = 1, 2 * params.layers do
    model.start_s[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.ds[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end
  model.core_network = core_network
  model.rnns = g_cloneManyTimes(core_network, params.seq_length)
  model.norm_dw = 0
  model.err = transfer_data(torch.zeros(params.seq_length))
  return model
end

function reset_state(state, model)
  state.pos = 1
  state.sent = 1
  state.begin = 1
  state.en = 1
  if model ~= nil and model.start_s ~= nil then
    model.start_d:zero()
    for d = 1, 2 * params.layers do
      model.start_s[d]:zero()
    end
  end
end

function reset_ds(model)
  model.dd:zero()
  for d = 1, #model.ds do
    model.ds[d]:zero()
  end
end

local function reset_s(model)
	for j = 0, params.seq_length do
  	  model.s[j] = {}
  	  model.d[j] = transfer_data(torch.zeros(params.batch_size, params.cond_len))
  	  for d = 1, 2 * params.layers do
  	    model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  	  end
  	end
	--model.s[0]:zero()
	--model.d[0]:zero()
end

-- state.pos tracks where in data we are traiing at the momeht 
function fp(model, feats, sequence, mask)
  reset_s(model)
  local d_err = 0
  model.d[0] = feats:transpose(1,2) -- gModule expects row wise examplesd
  for i = 1,sequence:size(1)-1 do
    local x = sequence[i]
	local y = sequence[i + 1]
    local s = model.s[i - 1]
	local d = model.d[i - 1]

    model.err[i], model.s[i], model.d[i] = unpack(model.rnns[i]:forward({x, y, s, d}))

	if params.obj_dnorm then
		local mask_diff = mask[i] - mask[i+1]
		local mask_diff = mask_diff:reshape(mask_diff:size(1),1):repeatTensor(1,params.cond_len)
		local d_tensor = torch.abs(torch.cmul(mask_diff, model.d[i]))
		d_err = d_err + d_tensor:sum()
	end
  end
  --model.start_d:copy(model.d[state.en]) 
  errs = model.err:sub(1,sequence:size(1)-1)
  --print(errs)
  if params.obj_dnorm then
	return errs:sum() + d_err/params.batch_size
  else
	return errs:sum()
  end
end

function bp(model, feats, sequence, mask)
  paramdx:zero()
  reset_ds(model)
  local mask_track = transfer_data(torch.zeros(params.batch_size,1))
  for i = sequence:size(1)-1,1,-1 do
    local x = sequence[i]
	local y = sequence[i + 1]
    local s = model.s[i - 1]
    local d = model.d[i - 1]

	local derr = transfer_data(torch.ones(1))

	if params.obj_dnorm then
		local mask_diff = mask[i] - mask[i+1]
		local mask_diff = mask_diff:reshape(mask_diff:size(1),1):repeatTensor(1,params.cond_len)
		mask_diff:cmul(torch.sign(model.d[i]))
		model.dd = model.dd + mask_diff
	end

    local tmp = model.rnns[i]:backward({x, y, s, d},
    	                                   {derr, model.ds, model.dd})
    g_replace_table(model.ds, tmp[3])
    model.dd:copy(tmp[4])
    cutorch.synchronize()
	i = i - 1
  end
  --state.pos = state.pos + params.seq_length
  --reset state.pos so that we can proceed with fp(state)

  if not params.grad_checking then
	model.norm_dw = model.paramdx:norm()
  	if model.norm_dw > params.max_grad_norm then
  	  local shrink_factor = params.max_grad_norm / model.norm_dw
	  if not params.custom_optimizer then --Vanilla GD
		model.paramdx:mul(shrink_factor)
	  end
  	end
	if not params.custom_optimizer then --Vanilla GD
	  model.paramx:add(model.paramdx:mul(-params.lr))
    end
  end
end

function run(train_test_valid,epoch)
  --reset_state(state)
  g_disable_dropout(model.rnns)
  local len = data:get_batch_count(train_test_valid)
  local perp = 0

  local errors = {}
  local data_ids = {}
  if params.err_display then io.output(io.stdout) end
  for i = 1, len do
    local sequences = data:get_next_batch(train_test_valid)
	local err = fp(model,sequences[2],sequences[1],sequences[3])
	table.insert(errors,err)
	table.insert(data_ids,sequences[4])
    perp = perp + err
	if params.err_display then io.write(g_d(err)) io.write(' ') end
  end
  if params.run_log[data:get_id(train_test_valid)] == 1 then 
	  local path = save_path .. train_test_valid .. '/errors_analysis/'
	  os.execute("mkdir -p " .. path)
	  local fp = io.open(path .. tostring(epoch),'w')
	  for i = 1,#errors do
		  local ids = data_ids[i]
		  for _,id in ipairs(ids) do
			  fp:write(data.dact[id],'\n')
			  fp:write(data.raw_utt[id],'\n')
		  end
		  fp:write(errors[i],'\n\n')
	  end
  end
  if params.err_display then io.write('\n') end
  print(train_test_valid .. " set error : " .. g_f3(perp / len))
  
  g_enable_dropout(model.rnns)
  return perp/len
end	

function run_valid_cond(train_test_valid,epoch,valid_sampler)
	local generated_sentences, generated_sentence_d
	--generated_sentences, generated_sentences_d = valid_sampler:sample_fp(state.uniq.feats)
	generated_sentences,_,_,gen_ids,gen_delex = valid_sampler:sample_fp(train_test_valid)
	valid_sampler:writeToFile(train_test_valid,generated_sentences,gen_delex,gen_ids,epoch)
	--gen_sent_lex = valid_sampler:gen_lex(generated_sentences,state.uniq,epoch) valid_sampler:writeToFile(train_test_valid,generated_sentences,gen_delex,gen_ids,epoch)
	--plot_rdgate(128,gen_sent_lex,generated_sentences,generated_sentences_d)
	
	if params.compute_bleu then
		local bleu = compute_bleu(gen_delex,train_test_valid,epoch)
		return bleu
	else
		return nil
	end
end

local function data_load()
--	save_path = './results1/' .. tostring(params.input) .. '_' .. tostring(epoch_counter) .. '/'

	data = data(params)

	if params.bleu_ref then
		data:bleu_ref_write()
	end

	params.cond_len   = data.feat_len
	params.vocab_size = data.vocab_size

	print('Data Loaded')

	--torch.save('../models/islot_index',islot_index)
	--torch.save('../models/vocab_map',vocab_map)
	--torch.save('../models/vocab_imap',vocab_imap)
	--torch.save('../models/slot_number',slot_number)
	--torch.save('../models/slot_number_',slot_number_)
	--torch.save('../models/dact_number',dact_number)
	--torch.save('../models/dact_number_',dact_number_)
end

local function feval(x) 
	if x ~= paramx then
		paramx:copy(x)
	end
	perp = fp(state_train)
	bp(state_train)

	return perp, paramdx
end

function compute_bleu(generated_sentences,train_test_valid,epoch)
	--local bleu_ref = {}
	local dact_rep
	local bleu_eval = {}
	local i
	local count = 1
	--local size = state.data:size(2)
	local len = data:get_uniq_batch_count(train_test_valid,1)
	--for s,sent in ipairs(generated_sentences) do
	for sent_id = 1,len do
		--i = state.ids[math.ceil(s/size)]:sub(1,1,count,count)[1][1]
		local sequences = data:get_next_uniq_batch(train_test_valid,1)
		local ids = sequences[4]
		i = ids[1]
		if bleu_eval[data.dact_delex_rep_str[i]] == nil then
			bleu_eval[data.dact_delex_rep_str[i]] = {}
		end
		local sent = generated_sentences[sent_id]
	  	table.insert(bleu_eval[data.dact_delex_rep_str[i]],sent)
	end

	--file = io.open(save_path..'bleu_ref','w')
	--io.output(file)
	--for dact_rep,utts in pairs(bleu_ref) do
	--	io.write(dact_rep)
	--	io.write('\n')
	--	for _,utt in pairs(utts) do
  	--	    io.write(utt)
	--		io.write('\n')
  	--	end
	--	io.write('\n')
  	--end
	--io.close(file)

	local full_path = params.save_path .. train_test_valid
	local file = io.open(full_path .. '/bleu_eval/'..tostring(epoch),'w')
	for dact_rep,utts in pairs(bleu_eval) do
		file:write('*',dact_rep,'\n')
		for _,utt in pairs(utts) do
			if params.bwd then	file:write(reverse(utt))
			else				file:write(utt)
			end
			file:write('\n')
  		end
		file:write('\n')
  	end
	file:close()

	os.execute('python bleu.py ' ..full_path.. '/ ' .. tostring(epoch))
	local file = io.open(full_path .. '/bleu/' .. tostring(epoch),'r')
	local bleu_score = tonumber(file:read())
	file:close()
	return bleu_score
end
----------------------------------------------------
								

local function run_flow()--(save_path,input_size,max_epoch)

	local optim_config
	if		opt.optimizer=='rmsprop' then 	optim_config = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
	elseif	opt.optimizer=='adagrad' then	optim_config = {learningRate = opt.learning_rate}
	elseif	opt.optimizer=='sgd'	 then 	optim_config = {learningRate = opt.learning_rate, momentum = opt.momentum}
	elseif	opt.optimizer=='adam'	 then 	optim_config = {learningRate = opt.learning_rate}
	else 	error('undefined optimizer')
	end

	print("Network parameters:")
	--print(params)
	--local states = {state_train, state_valid, state_test}
	--for _, state in pairs(states) do
	--  reset_state(state)
	--end
	local step = 0
	local epoch = 0
	local total_cases = 0
	local beginning_time = torch.tic()
	local start_time = torch.tic()
	print("Starting training.")
	local words_per_step = params.seq_length * params.batch_size

	local epoch_size = data:get_batch_count('train')
	print('epoch_size',epoch_size)
	
	local train_errors = {}
	local errors = {{},{},{}}
	local bleus = {{},{},{}}
	
	model = setup()
	
	local perps

	while epoch < params.max_max_epoch do

		local perp
		if params.custom_optimizer then
			if		opt.optimizer=='rmsprop' then 	_, loss = optim.rmsprop(feval, paramx, optim_config)
			elseif	opt.optimizer=='adagrad' then	_, loss = optim.adagrad(feval, paramx, optim_config)
			elseif	opt.optimizer=='sgd'	 then	_, loss = optim.sgd(feval, paramx, optim_config)
			elseif	opt.optimizer=='adam'	 then	_, loss = optim.adam(feval, paramx, optim_config)
			else	error('undefined optimizer')
			end
			perp = loss[1]
		else
			local sequences = data:get_next_batch('train')
			perp = fp(model,sequences[2],sequences[1],sequences[3])
			bp(model,sequences[2],sequences[1],sequences[3])
		end

		if perps == nil then
			perps = torch.zeros(epoch_size):add(perp)
		end
		perps[step % epoch_size + 1] = perp
		step = step + 1
		total_cases = total_cases + params.seq_length * params.batch_size
		epoch = step / epoch_size
		if step % torch.round(epoch_size * params.stats_freq) == 0 then
			local wps = torch.floor(total_cases / torch.toc(start_time))
	    	local since_beginning = g_d(torch.toc(beginning_time) / 60)
	    	      --', train perp. = ' .. g_f3(torch.exp(perps:mean())) ..
			io.output(io.stdout)
	    	print('epoch = ' .. g_f3(epoch) ..
	    	      ', train error = ' .. g_f3(perps:mean()) ..
	    	      ', wps = ' .. wps ..
	    	      ', dw:norm() = ' .. g_f3(model.norm_dw) ..
	    	      ', lr = ' ..  g_f3(params.lr) ..
	    	      ', since beginning = ' .. since_beginning .. ' mins.')
		end
		if step % epoch_size == 0 then
			table.insert(train_errors,perps:mean())
			if params.lr_decay and (epoch > params.max_epoch) then
				params.lr = params.lr / params.decay
	    	end
		end

		if params.log_err and (step > 0) and (step % (epoch_size*params.log_err_freq) == 0) then
			--table.insert(valid_errors,run_valid())
			if params.err_display then
				for t = 1,perps:size(1) do io.write(g_d(perps[t])) io.write(' ') end
				io.write('\n')
			end
			for i = 1,3 do if params.err_log[i] then table.insert(errors[i],run(TRAIN_VAL_TEST[i],epoch)) end end
	  	end

		if params.log_bleu and (step > 0) and (step % (params.log_bleu_freq*epoch_size) == 0) then
			local valid_sampler = sampler(params,sampling_params,model.paramx,nil,data)
			for i = 1,3 do
				if params.bleu_log[i] then
					local test_bleu = run_valid_cond(TRAIN_VAL_TEST[i],epoch,valid_sampler)
					table.insert(bleus[i],test_bleu)
				end
			end
	  	end
	  	if step % 33 == 0 then
			cutorch.synchronize()
			--collectgarbage()
	  	end
	  	if params.model_checkpoint and (step > 0) and ((step % (params.model_checkpoint_freq*epoch_size)) == 0) then
			if params.bwd then
				torch.save(save_path .. 'models/bwd_model' .. tostring(epoch),model.paramx);
			else
				torch.save(save_path .. 'models/fwd_model' .. tostring(epoch),model.paramx);
			end
	  	end
	end

	--if params.log_dir then
	--	if params.bwd then
	--		torch.save(save_path .. 'models/bwd_model',model);
	--	else
	--		--torch.save(save_path .. 'models/fwd_model',model);
	--		torch.save(save_path .. '/params',model.paramx);
	--		torch.save(save_path .. '/run_params',params);
	--		torch.save(save_path .. '/sampling_params',sampling_params);
	--		local data_properties = {
	--			vocab_map			= data.vocab_map,
	--			ivocab_map			= data.ivocab_map,
	--			START_SYMBOL		= data.START_SYMBOL,  
	--			STOP_SYMBOL			= data.STOP_SYMBOL,		   
	--			islot_index			= data.islot_index,		   
	--			NSSacts				= data.NSSacts,			   
	--			system_actslot_inds	= data.system_actslot_inds
	--		}
	--		torch.save(save_path .. 'models/data_properties',data_properties);
	--	end
	--end

	if params.make_plots then
		plots(train_errors,'Training Error')
		for i = 1,3 do
			if params.err_log[i]  then plots(errors[i],TRAIN_VAL_TEST[i] .. ' Error') end
			if params.bleu_log[i] then plots(bleus[i],TRAIN_VAL_TEST[i] .. ' (BLEU)') end
		end
	end
end

local function generate_test()

	local fwd_model
	if params.run_flow then
		fwd_model = model --Running after training, so directly take trained model
	else
		fwd_model = torch.load(params.pre_init_mdl_path) -- Running function in isolation, so load
	end
	if params.bwd_rerank then
		local bwd_model = torch.load(save_path .. params.pre_init_bwdmdl_path)
		sampler = sampler(params,sampling_params,fwd_model,bwd_model,data)
	else
		sampler = sampler(params,sampling_params,fwd_model,nil,data)
	end

	local train_test_valid = 'test'
	local generated_sentences,_,_,gen_ids,gen_delex = sampler:sample_fp(train_test_valid)
	sampler:writeToFile(train_test_valid,generated_sentences,gen_delex,gen_ids,'samples')
	local bleu = compute_bleu(gen_delex,train_test_valid,'samples')

	print('Test BLEU score: ', bleu)
	
end

local function interact()

	local fwd_model = torch.load(save_path .. 'models/fwd_model')
	local bwd_model = torch.load(save_path .. 'models/bwd_model')
	local n = sampling_params.overgen
	local m = 1

	local interact_sampler = sampler(params,sampling_params,model,nil,data)
	io.input(io.stdin)
	io.write("Enter input\n")
	io.flush()
	input = io.read()
	--print(input)
	repeat
		local _,_,_,test_dact,test_dacts_sv = parse_data({input})
		local dact_1hot,_ = dact_features(Nacts,Nslots,test_dact,test_dact,test_dacts_sv,dact_number,slot_number)
		if dact_1hot then
			dact_1hot = transfer_data(torch.reshape(dact_1hot,params.cond_len,1))
			--print(dact_1hot)
			--sample_setup(model)
			--generated_sentences, generated_sentences_d = sample_fp(dact_1hot,0)
			--state_test = {feats=transfer_data(dact_1hot)}
			n = 10
			m = 10

			--gen_sent = sample_n(params,dact_1hot,fwd_model,bwd_model,test_dacts_sv[1],n,m)
			print(gen_sent)
			--gen_sent_lex = lex(gen_sent[1],dacts_sv)
			--print(gen_sent_lex[1])
		else
			print("Expression Invalid")
		end
		input = io.read()
	until input == 'exit'
end

mkdirs()
data_load()

if params.batch_size > 1 then
	params.lr = params.lr_batch
else
	params.lr = params.lr_sgd
end
if params.data_debug_print then
	data:write_data_format()
end
if params.grad_checking then grad_check()	end
if params.run_flow		then run_flow()		end
if params.generate_test then generate_test() end
--interact()

