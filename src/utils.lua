function plots(errors,title)
	num_data = #errors
	local x = torch.linspace(1,num_data,num_data)
	local plot = {}
	gnuplot.title(title)
	gnuplot.xlabel('Epoch')
	--gnuplot.ylabel(ylabel)
	gnuplot.raw('set terminal pngcairo')
	gnuplot.raw('set pointsize 0')
	gnuplot.raw('set output "' .. save_path .. '/' .. title .. '.png"')
	gnuplot.plot({x,torch.Tensor(errors)})
	
end

function mkdirs()
	if params.log_dir then
		params.save_path = save_path
		os.execute("mkdir " .. save_path)
		os.execute("mkdir " .. save_path .. 'models')
		if params.log_bleu then
			os.execute("mkdir -p " .. save_path .. 'train/gen')
			os.execute("mkdir -p " .. save_path .. 'train/bleu')
			os.execute("mkdir -p " .. save_path .. 'train/bleu_eval')
			os.execute("mkdir -p " .. save_path .. 'train/bleu_eval_scores')
			os.execute("mkdir -p " .. save_path .. 'val/gen')
			os.execute("mkdir -p " .. save_path .. 'val/bleu')
			os.execute("mkdir -p " .. save_path .. 'val/bleu_eval')
			os.execute("mkdir -p " .. save_path .. 'val/bleu_eval_scores')
		end
		if params.generate_test then
			os.execute("mkdir -p " .. save_path .. 'test/gen')
			os.execute("mkdir -p " .. save_path .. 'test/bleu')
			os.execute("mkdir -p " .. save_path .. 'test/bleu_eval')
			os.execute("mkdir -p " .. save_path .. 'test/bleu_eval_scores')
		end

		param_file = io.open(save_path .. "params","w")
		io.output(param_file)
		for key, value in pairs (params) do
		    io.write(string.format("[%s] : %s\n",
		    tostring (key), tostring(value)))
		end
		io.close(param_file)
		param_file = io.open(save_path .. "sampling_params","w")
		io.output(param_file)
		for key, value in pairs (params) do
		    io.write(string.format("[%s] : %s\n",
		    tostring (key), tostring(value)))
		end
		io.close(param_file)
		io.output(io.stdout)
	end
end

--function compute_bleu(generated_sentences,state,epoch)
--	--local bleu_ref = {}
--	local dact_rep
--	local bleu_eval = {}
--	local i
--	local count = 1
--	local size = state.data:size(2)
--	for s,sent in ipairs(generated_sentences) do
--		--i = state.ids[math.ceil(s/size)]:sub(1,1,count,count)[1][1]
--		i = state.ids[math.ceil(s/size)][count]
--		count = count + 1
--		if count > size then
--			count = 1
--		end
--		if bleu_eval[data.dact_delex_rep_str[i]] == nil then
--			bleu_eval[data.dact_delex_rep_str[i]] = {}
--		end
--	  	table.insert(bleu_eval[data.dact_delex_rep_str[i]],sent)
--	end
--
--	--file = io.open(save_path..'bleu_ref','w')
--	--io.output(file)
--	--for dact_rep,utts in pairs(bleu_ref) do
--	--	io.write(dact_rep)
--	--	io.write('\n')
--	--	for _,utt in pairs(utts) do
--  	--	    io.write(utt)
--	--		io.write('\n')
--  	--	end
--	--	io.write('\n')
--  	--end
--	--io.close(file)
--
--	local full_path = params.save_path .. state.name 
--	local file = io.open(full_path .. '/bleu_eval/'..tostring(epoch),'w')
--	io.output(file)
--	for dact_rep,utts in pairs(bleu_eval) do
--		io.write('*')
--		io.write(dact_rep)
--		io.write('\n')
--		for _,utt in pairs(utts) do
--			if params.bwd then
--				io.write(reverse(utt))
--			else
--				io.write(utt)
--			end
--			io.write('\n')
--  		end
--		io.write('\n')
--  	end
--	io.close(file)
--	io.output(stdout)
--
--	os.execute('python bleu.py ' ..full_path.. '/ ' .. tostring(epoch))
--	local file = io.open(full_path .. '/bleu/' .. tostring(epoch))
--	io.input(file)
--	local bleu_score = tonumber(io.read())
--	io.close(file)
--	io.output(io.stdout)
--	return bleu_score
--end


