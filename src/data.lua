require 'utils.concat'
require 'cunn'
JSON = (loadfile "JSON.lua")() 
local stringx = require('pl.stringx')
local file = require('pl.file')
--package.path = package.path .. ';../../?.lua'
require('data_helpers')

function transfer_data(x)
	return x:cuda()
end

local data = torch.class('data')
function data:__init(params)
	self.organize_mask_format = true

	self.dact		= nil		--Dialog acts (string)
	self.raw_utt	= nil		--Raw utterances (string)
	self.sent		= nil		--sent[dact] = {utterances corresponding to dact}
	self.Ndata		= nil		--Number of data instances

	self.acts		= nil		--acts[act] = true
	self.slots		= nil		--slots[slot] = true
	self.values		= nil		--values[slot][value] = true
	self.dacts_sv	= nil		--dacts_sv[i][slot] = value
	self.dacts		= nil		--dacts[i] = dialog act

	self.dact_number	= nil	--dact -> idx
	self.idact_number	= nil	--idx  -> dact
	self.slot_number	= nil	--slot -> idx 
	self.islot_number	= nil	--idx  -> slot
	self.slot_index		= nil   --slot -> 'value' + slot_idx
	self.islot_index	= nil	--reverse of above
	self.Nacts			= nil   	
	self.Nslots			= nil

	self.raw_utt_delex	= nil	--[i] = delexicalized utterance
	self.raw_utt_token	= nil 	--[i] = delexicalized utterance tokens
	self.lengths		= nil 	--[i] = lenths of delexicalized utterances

	self.features	= nil		--[i] = feature vector
	self.feat_len	= nil		--length of feature vectors

	self.vocab_map	= nil		--word -> idx
	self.ivocab_map = nil		--idx  -> word
	self.x			= nil		--[i]  -> '1' + word indices + 'w'
	self.vocab_size = nil

	self.train_inds = nil		--Indices
	self.valid_inds = nil
	self.test_inds  = nil

	self.write_train_valid_test = true

	self.data = nil

	self.batch_size = params.batch_size
	self.unique_instances = params.data_unique_instances
	self.data_split_ratio = params.data_split_ratio
	self.params = params

	self.dact_delex_rep_str = nil --Sorted string rep of dacts Eg:'2,1,3'

	self.official_split = false

------------------------------------------

	self.system_acts_noarg	= {'affirm','bye','canthear','confirm-domain','negate','repeat','reqmore','welcomemsg'}
	self.system_acts_arg	= {'canthelp','canthelp.missing_slot_value','expl-conf','impl-conf','inform','offer','request','select','canthelp.exception'}
	--self.system_acts_arg	= {'canthelp.missing_slot_value','expl-conf','impl-conf','inform','offer','request','select'}
	self.system_acts		= {'affirm','bye','canthear','confirm-domain','negate','repeat','reqmore','welcomemsg','canthelp','canthelp.missing_slot_value','expl-conf','impl-conf','inform','offer','request','select'}

	self.allacts = {'affirm','bye','canthear','confirm-domain','negate','repeat','reqmore','welcomemsg','canthelp','canthelp.missing_slot_value','expl-conf','impl-conf','inform','offer','request','select','ack','hello','null','reqalts','restart','silence','thankyou','confirm','deny','canthelp.exception'}

	self.slots = {'area','food','name','pricerange','addr','phone','postcode','signature','count','area_dontcare','food_dontcare','pricerange_dontcare','phone_dontcare'}
	
	local i 
	self.slot_index = {}
	self.islot_index = {}
	for i = 1,#self.slots do
		self.slot_index[self.slots[i]] = i
		self.islot_index['value' .. tostring(i)] = self.slots[i]
	end

	self.user_act_inds		= {}
	self.system_act_inds	= {}
	self.allact_inds		= {}
	self.slot_inds			= {}

	--for i = 1,#self.user_acts	do self.user_act_inds[self.user_acts[i]]		= i end
	for i = 1,#self.system_acts	do self.system_act_inds[self.system_acts[i]]	= i end
	for i = 1,#self.allacts		do self.allact_inds[self.allacts[i]]			= i end
	for i = 1,#self.slots		do self.slot_inds[self.slots[i]]				= i end


	self.system_actslot_inds = {}
	self.isystem_actslot_inds = {}
	for i = 1,#self.system_acts_noarg do
		self.system_actslot_inds[self.system_acts_noarg[i]] = i
		self.isystem_actslot_inds[i] = self.system_acts_noarg[i]
	end
	ct = #self.system_acts_noarg + 1
	for i = 1,#self.system_acts_arg do
		for j = 1,#self.slots do
			self.system_actslot_inds[self.system_acts_arg[i] .. self.slots[j]] = ct
			self.isystem_actslot_inds[ct] = self.system_acts_arg[i] .. self.slots[j]
			ct = ct + 1
		end
	end
	self.NSSacts = ct - 1
	self.feat_len = self.NSSacts

------------------------------------------------------------------

	--self.state_train		= nil --{data=cuda_tensor(sentences x batch_size), feats=cuda_tensor(feats x batch_size), ids=(num_sentences x batch_size)
	--self.state_train.uniq	= nil 
	--self.state_valid		= nil 
	--self.state_valid.uniq	= nil 
	--self.state_test			= nil 
	--self.state_test.uniq	= nil 

	self.data_ptr = {1,1,1}
	self.alldata = {{},{},{}}

	self:read_data()
	--self:parse_data()
	--self:get_mappings()
	self:delexicalize()
	self:dact_features()
	self:vocab()
	self:dact_delex_rep()
	self:determine_split()
	--self:organize_data()
	--self:bleu_ref(0)


end

function data:vocab(raw_utt_delex,dact)

	local raw_utt_delex = self.raw_utt_delex
	local dact			= self.dact

	local vocab_idx = 2
	local vocab_map = {}
	local ivocab_map = {}

	x = {}
	self.START_SYMBOL = 1
	self.STOP_SYMBOL = 2
	ivocab_map[1] = 'START'
	ivocab_map[2] = 'STOP'
	for i = 1, #dact do
		x[i] = {}
		--print(raw_utt_delex[i])
		utterance = string.split(raw_utt_delex[i]," ")
		for j = 1, #utterance do        
			if vocab_map[utterance[j]] == nil then
				vocab_idx = vocab_idx + 1
				vocab_map[utterance[j]] = vocab_idx
				ivocab_map[vocab_idx] = utterance[j]
			end
			if bwd then
				x[i][#utterance+2-j] = vocab_map[utterance[j]]
			else
				x[i][j+1] = vocab_map[utterance[j]]
			end
		end
		x[i][1] = 1 -- START SYMBOL
		x[i][#utterance+2] = 2 -- STOP SYMBOL 
	end

	--return vocab_map, ivocab_map, x
	self.vocab_map = vocab_map   --word -> idx
	self.ivocab_map = ivocab_map --idx  -> word
	self.x = x					 --[i]  -> '1' + word indices + 'w'
	self.vocab_size = vocab_idx
end	

function data:dact_features()

	--local feat_len = self.Nacts
	--for _,_ in pairs(self.slot_number) do
	--	feat_len = feat_len + 1
	--end
	local features = torch.zeros(self.Ndata,self.feat_len)
	local features_table = {}

	for i = 1, self.Ndata do

		local sys, dacts_sv_perturn, dact_sv_string, act_table = self:get_tensor_system(self.dialog_acts[i])
		features[{{i},{}}] = sys:reshape(1,sys:size(1))
		table.insert(features_table, act_table)
		--if self.dact_number[self.dacts[i]] == nil then return nil end
		--features[i][self.dact_number[self.dacts[i]]] = 1
		--local slot, value
		--for slot,value in pairs(self.dacts_sv[i]) do
		--	if self.params.dont_care_handle and ((value == 'dont_care') or (value == 'yes') or (value == 'no')) then
		--		features[i][self.Nacts + self.slot_number[slot..'_'..value]] = 1
		--	end
		--	assert(self.slot_number[slot] ~= nil, "nil slot")
		--	features[i][self.Nacts + self.slot_number[slot]] = 1
		--end
	end

	self.features = features
	self.features_table = features_table
	--self.feat_len = feat_len
	--return features, feat_len
end

function data:features_to_dactstring(feature_tensor)

	local dasv_table = {}
	local dact_string = ''
	for i = 1,#self.system_acts_noarg do
		if feature_tensor[i] == 1 then
			table.insert(dasv_table,{['dialog-act'] = self.system_acts_noarg[i]})
			dact_string = dact_string .. self.system_acts_noarg[i] .. '() ' 
		end
	end
	local ct = #self.system_acts_noarg + 1
	for i = 1,#self.system_acts_arg do
		for j = 1,#self.slots do
			if feature_tensor[ct] == 1 then 
				table.insert(dasv_table,{['dialog-act'] = self.system_acts_arg[i],slot = self.slots[j]})
				dact_string = dact_string .. self.system_acts_arg[i] .. '(' .. self.slots[j] .. ') ' 
			end
			ct = ct + 1
		end
	end

	return dact_string
end	

function data:get_id(train_test_valid)
	if train_test_valid == 'train'		then return 1
	elseif train_test_valid == 'val'	then return 2
	elseif train_test_valid == 'test'	then return 3
	else
		print(train_test_valid)
		assert('Invalid input')
	end
end

function data:debug_print()
	local file = io.open('debug_print','w')
	for i = 1,self.Ndata do
		file:write(self.dact[i],'\n')
		local dact_tensor = self.features[i]
		local len = dact_tensor:size(1)
		for j = 1,len			do file:write(dact_tensor[j],' ') end file:write('\n')
		for j = 1,self.Nacts	do file:write(self.idact_number[j],' ',dact_tensor[j],' ') end file:write('\n')
		for j = 1,self.Nslots	do file:write(self.islot_number[j],' ',dact_tensor[self.Nacts + j],' ') end file:write('\n\n')
		for j = 1,#self.x[i]	do file:write(self.x[i][j],' ') end file:write('\n\n')
	end
	file:close()
end

function data:read_data(tvt)

	--path = '/home/llajan/data'
	path = '../../data'
	dir_prefix = {	    path .. '/train/',
						path .. '/train/',
						path .. '/test/'}
	local train = read_file(path .. '/config/dstc2_train.flist')
	local valid = read_file(path .. '/config/dstc2_dev.flist')
	local test  = read_file(path .. '/config/dstc2_test.flist')

	local alldialogs = {train,valid,test}

	local system_dialog_tensor = nil
	local raw_utt = {}
	local dact_sv = {}
	local tensor_list = {}
	local dacts_sv_string = {}
	local dialog_acts = {}
	local dact = {}
	local ct = 1
	for tvt = 1,3 do
		--print(tvt)
		local unique_track = {}
		for i = 1,#alldialogs[tvt] do

			local user = read_json	(dir_prefix[tvt] .. alldialogs[tvt][i] .. '/label.json')
			local system = read_json	(dir_prefix[tvt] .. alldialogs[tvt][i] .. '/log.json')

			local turns = #user['turns']
			for j = 1,turns do
				local system_acts = system['turns'][j]['output']['dialog-acts']
				local sentence = system['turns'][j]['output']['transcript']

				local sys, dacts_sv_perturn, dact_sv_string = self:get_tensor_system(system_acts)

				local unique_flag = true
				if unique_track[dact_sv_string .. sentence] == nil then
					unique_track[dact_sv_string .. sentence] = true
					unique_flag = false
				end

				if (self.unique_instances and unique_flag) or (not self.unique_instances) then
					--print(dact_sv_string)
					--for i = 1, sys:size(1) do
					--	if sys[i] == 1 then
					--		io.write(self.isystem_actslot_inds[i], ' ')
					--	end
					--end
					--io.write('\n')

					
					--table.insert(dacts_sv_string, dact_sv_string)
					--table.insert(tensor_list,sys)
					--
					--local sys_hor = sys:reshape(1,sys:size(1))
					--if system_dialog_tensor == nil then system_dialog_tensor = sys_hor
					--else system_dialog_tensor = torch.cat(system_dialog_tensor,sys_hor,1)
					--end


					dact[ct] = dact_sv_string
					dact_sv[ct] = dacts_sv_perturn
					table.insert(raw_utt,sentence)
					table.insert(dialog_acts,system_acts)
					table.insert(self.alldata[tvt],ct)
					ct = ct + 1
				end
			end
		end
	end
	print(#self.alldata[1],#self.alldata[2],#self.alldata[3])
	self.Ndata = #raw_utt
	self.raw_utt = raw_utt
	self.dialog_acts = dialog_acts
	self.dact = dact
	self.dacts_sv = dact_sv
	--self:delexicalize()
	--print('Dataset size ') print(#dacts_sv_string)
	--return {utterances=utterances, utterances_delex=utterances_delex, dacts_sv=dact_sv, tensors=system_dialog_tensor, tensor_list=tensor_list, dacts_sv_string}
end

function data:delexicalize()
	local delex_utterances = {}
	for i = 1,self.Ndata do
		local utterance = self.raw_utt[i]
		--print(utterance)
		local dact_sv = self.dacts_sv[i]
		for s,values in pairs(dact_sv) do
			--print(utterance)
			--print(val)
			---- hyphen is a magic character. Escape it.
			--if #values > 1 then print(utterance) end
			for _,val in ipairs(values) do
				if string.find(val,'-') then 
					u,v = string.find(utterance:lower(),val:lower():gsub("[%-]","%%%0"))
				else
					u,v = string.find(utterance:lower(),val:lower())
				end
				if u ~= nil then
					before = string.sub(utterance,u-1,u-1)
					after = string.sub(utterance,v+1,v+1)
					if not(string.match(before,'%w') or string.match(after,'%w')) then
						--print(value)
						utterance = string.sub(utterance,1,u-1) .. 'value' .. self.slot_index[s] .. string.sub(utterance,v+1)
					end
				--else
				--	print(val)
				end
			end
			--if #values > 1 then print(utterance) end
		end
		table.insert(delex_utterances,utterance)
	end
	self.raw_utt_delex = delex_utterances
end

function data:vocab()

	local raw_utt_delex = self.raw_utt_delex

	local vocab_idx = 2
	local vocab_map = {}
	local ivocab_map = {}

	x = {}
	self.START_SYMBOL = 1
	self.STOP_SYMBOL = 2
	ivocab_map[1] = 'START'
	ivocab_map[2] = 'STOP'
	--for i = 1, 10 do
	for i = 1, self.Ndata do
	    x[i] = {}
		--print(raw_utt_delex[i])
	    utterance = string.split(raw_utt_delex[i]," ")
	    for j = 1, #utterance do        
	        if vocab_map[utterance[j]] == nil then
	            vocab_idx = vocab_idx + 1
	            vocab_map[utterance[j]] = vocab_idx
				ivocab_map[vocab_idx] = utterance[j]
	        end
			if bwd then
				x[i][#utterance+2-j] = vocab_map[utterance[j]]
			else
				x[i][j+1] = vocab_map[utterance[j]]
			end
	    end
	    x[i][1] = 1 -- START SYMBOL
	    x[i][#utterance+2] = 2 -- STOP SYMBOL 
	end

	--for i = 1,self.Nacts do
	--	vocab_idx = vocab_idx + 1
	--	vocab_map['dact_' .. self.idact_index[i]] = vocab_idx
	--	ivocab_map[vocab_idx] = 'dact_' .. self.idact_index[i]
	--end

	--vocab_idx = vocab_idx + 1
	--ivocab_map[vocab_idx] = 'presence of slot'
	--vocab_map['presence of slot'] = vocab_idx

	--vocab_idx = vocab_idx + 1
	--ivocab_map[vocab_idx] = 'absence of slot'
	--vocab_map['absence of slot'] = vocab_idx

	--vocab_idx = vocab_idx + 1
	--ivocab_map[vocab_idx] = 'dont_care'
	--vocab_map['dont_care'] = vocab_idx

	--return vocab_map, ivocab_map, x
	self.vocab_map = vocab_map   --word -> idx
	self.ivocab_map = ivocab_map --idx  -> word
	self.x = x					 --[i]  -> '1' + word indices + 'w'
	self.vocab_size = vocab_idx
end	

function data:determine_split()

	local train_inds,train_inds_uniq = {}, {}
	local valid_inds,valid_inds_uniq = {}, {}
	local test_inds, test_inds_uniq	 = {}, {}

	if not self.official_split then
		--The following code equalizes the distribution of dacts
		--Adding examples to each type of dact till we have 20
		--thresh = 20
		--for u,v in pairs(bleu_ref_ids) do
		--	local len = #v
		--	if #v < thresh then 
		--		for ct = 1,thresh-#v do
		--			table.insert(bleu_ref_ids[u],v[(ct % #v) + 1])
		--		end
		--	end
		--end

		local train_inds_uniq_marker = {}
		local valid_inds_uniq_marker = {}
		local test_inds_uniq_marker	 = {}
		local ratio = self.data_split_ratio
		local ref_dact
		for ref_dact, dact_ids in pairs(self.delex_act_ids) do
			local total = #dact_ids
			--Ensure same ratio within each dact class
			local num_train = math.ceil(ratio[1]*total)
			local num_valid = math.ceil((ratio[1] + ratio[2])*total)
			
			for i = 1,num_train do						
			--for i = 1,1 do						
				table.insert(train_inds,dact_ids[i])
				if train_inds_uniq[dact_ids[i]] == nil then 
					train_inds_uniq_marker[dact_ids[i]] = true
					table.insert(train_inds_uniq,dact_ids[i])
				end
			end
			local start = num_train
			if start < num_valid then start = start + 1 end
			for i = start,num_valid do 	
			--for i = 1,1 do						
				table.insert(valid_inds,dact_ids[i])
				if valid_inds_uniq[dact_ids[i]] == nil then
					valid_inds_uniq_marker[dact_ids[i]] = true
					table.insert(valid_inds_uniq,dact_ids[i])
				end
			end
			local start = num_valid
			if  start < total then	start = start + 1 end
			for i = start,total do		
			--for i = 1,1 do						
				table.insert(test_inds,dact_ids[i])
				if test_inds_uniq[dact_ids[i]] == nil then
					test_inds_uniq_marker[dact_ids[i]] = true
					table.insert(test_inds_uniq,dact_ids[i])
				end
			end
		end

	else
		train_inds		= self.alldata[1]
		valid_inds 		= self.alldata[2]
		test_inds  		= self.alldata[3]

		-- Find unique inds
		train_inds_uniq = self:find_unique(self.alldata[1])
		valid_inds_uniq = self:find_unique(self.alldata[2])
		test_inds_uniq  = self:find_unique(self.alldata[3])
	end
	

	train_inds		= self:round_inds(train_inds)
	valid_inds		= self:round_inds(valid_inds)
	test_inds		= self:round_inds(test_inds)
	train_inds_uniq = self:round_inds(train_inds_uniq)
	valid_inds_uniq = self:round_inds(valid_inds_uniq)
	test_inds_uniq  = self:round_inds(test_inds_uniq)

	if self.params.shuffle then
		self.train_inds		 = self:shuffle(train_inds)
		self.valid_inds 	 = self:shuffle(valid_inds)
		self.test_inds  	 = self:shuffle(test_inds)
		self.train_inds_uniq = self:shuffle(train_inds_uniq)
		self.valid_inds_uniq = self:shuffle(valid_inds_uniq)
		self.test_inds_uniq  = self:shuffle(test_inds_uniq)
	else
		self.train_inds		 = train_inds
		self.valid_inds 	 = valid_inds
		self.test_inds  	 = test_inds
		self.train_inds_uniq = train_inds_uniq
		self.valid_inds_uniq = valid_inds_uniq
		self.test_inds_uniq  = test_inds_uniq
	end

	if self.write_train_valid_test then
		self:writeDataToFile(self.train_inds,'./data/train_data')
		self:writeDataToFile(self.test_inds ,'./data/test_data')
		self:writeDataToFile(self.valid_inds,'./data/valid_data')
	end

	self.alldata = {self.train_inds, self.valid_inds, self.test_inds}
	self.alldata_uniq = {self.train_inds_uniq, self.valid_inds_uniq, self.test_inds_uniq}

end

function data:find_unique(inds)
	local track_unique = {}
	local uniq_inds = {}
	for _,ind in ipairs(inds) do
		local act = self.dact[ind]
		if not track_unique[act] then
			track_unique[act] = true
			table.insert(uniq_inds,ind)
		end
	end
	return uniq_inds
end

function data:round_inds(inds)
	local size = self.batch_size
	local inds_rounded = inds
	local count = 1
	while #inds_rounded % size ~= 0 do
		table.insert(inds_rounded,inds[count % #inds])
		count = count + 1
	end
	return inds_rounded
end

function data:shuffle(t)
	local n = #t
	while n >= 2 do
		local k = math.random(n)
		t[n], t[k] = t[k], t[n]
		n = n - 1
	end
	return t
end

function data:organize_mask(inds,size)

	local seq = self.x

	local batches_senoh = {}
	local sentence_ids = {}

	assert(#inds == size, 'Wrong number of inds')

	local max_len = 0
	for i = 1,size do
		local index = inds[i]
		if #seq[index] > max_len then max_len = #seq[index] end
	end

	batch_senoh = torch.zeros(max_len,size,self.vocab_size)
	batch_mask  = torch.zeros(max_len,size)
	batch_features = torch.zeros(self.feat_len,size)

	for i = 1,size do
		local index = inds[i]

		batch_features[{{},i}] = self.features[index]

		for ct = 1,#seq[index] do
			local word = seq[index][ct]
			batch_senoh[ct][i][word] = 1
			batch_mask[ct][i] = 1
		end
		batch_mask[#seq[index]][i] = 0 --This is so that the last instance where mask is 1 is when the output is STOP_SYMBOL
		table.insert(sentence_ids,index)
	end

	return {transfer_data(batch_senoh), transfer_data(batch_features), transfer_data(batch_mask), sentence_ids}
end

function data:reset_pointer(train_test_valid)
	self.data_ptr[self:get_id(train_test_valid)] = 1
end

function data:get_batch_count(train_test_valid)
	local tvt = self:get_id(train_test_valid)
	return #self.alldata[tvt]/self.batch_size
end

function data:get_uniq_batch_count(train_test_valid,sz)
	local tvt = self:get_id(train_test_valid)
	return #self.alldata_uniq[tvt]/sz
end

function data:get_next_uniq_batch(train_test_valid,sz)
	local tvt = self:get_id(train_test_valid)
	local ptr = self.data_ptr[tvt]
	local inds = {}
	for i = ptr, ptr + sz - 1 do
		table.insert(inds, self.alldata_uniq[tvt][i])
	end
	self.data_ptr[tvt] = self.data_ptr[tvt] + sz
	if self.data_ptr[tvt] > #self.alldata_uniq[tvt] then self.data_ptr[tvt] = 1 end
	local sequences = self:organize_mask(inds,sz)
	return sequences
end

function data:get_next_batch(train_test_valid)
	local tvt = self:get_id(train_test_valid)
	local ptr = self.data_ptr[tvt]
	local inds = {}
	for i = ptr, ptr + self.batch_size - 1 do
		table.insert(inds, self.alldata[tvt][i])
	end
	self.data_ptr[tvt] = self.data_ptr[tvt] + self.batch_size
	if self.data_ptr[tvt] > #self.alldata[tvt] then self.data_ptr[tvt] = 1 end
	local sequences = self:organize_mask(inds,self.batch_size)
	return sequences
	--return {sequences[1], sequences[2], sequences[3], sequences[4]}
end

function data:dact_delex_rep()

	local delex_act_sents = {}			--['dact str rep (Eg:1,2,3)'] -> {delexicalized utterance}
	local delex_act_dacts = {}		-- " -> {dialog act string}
	local delex_act_ids = {}		-- " -> {id}
	local bleu_ref = {}
	local bleu_ref_dacts = {}
	local bleu_ref_unique_marker = {}
	
	local dact_delex_rep_str = {}
	-- Group dacts into common delexicalized acts
	for i = 1,self.Ndata do
	    local dact_rep = {}
	    --table.insert(dact_rep,self.dact_number[self.dacts[i]])
		--for slot,value in pairs(self.dacts_sv[i]) do
		--	if (value == 'dont_care') then token = slot ..'_'..value end
		--	if care_values then table.insert(dact_rep,self.slot_number[token])
		--	else				table.insert(dact_rep,self.slot_number[slot]) end
		--end
		dact_rep = self.features_table[i]
	    table.sort(dact_rep)
	    local dact_rep_str = ''
	    for _,dact_rep_num in ipairs(dact_rep) do
	  	  dact_rep_str = dact_rep_str .. ',' .. tostring(dact_rep_num)
	    end
	
		dact_delex_rep_str[i] = dact_rep_str

	    if delex_act_sents[dact_rep_str] == nil then
	  	  delex_act_sents[dact_rep_str] = {}
		  delex_act_dacts[dact_rep_str] = {}
		  delex_act_ids[dact_rep_str] = {}
		  bleu_ref[dact_rep_str] = {}
		  bleu_ref_dacts[dact_rep_str] = {}
	    end
	    
		table.insert(delex_act_sents[dact_rep_str],self.raw_utt_delex[i])
		table.insert(delex_act_dacts[dact_rep_str],self.dact[i])
		table.insert(delex_act_ids[dact_rep_str],i) 

		if bleu_ref_unique_marker[dact_rep_str..self.raw_utt_delex[i]] == nil then
			table.insert(bleu_ref[dact_rep_str],self.raw_utt_delex[i])
			table.insert(bleu_ref_dacts[dact_rep_str],self.dact[i])
			bleu_ref_unique_marker[dact_rep_str..self.raw_utt_delex[i]] = true
		end
	end

	self.dact_delex_rep_str = dact_delex_rep_str
	self.delex_act_sents = delex_act_sents
	self.delex_act_dacts = delex_act_dacts
	self.delex_act_ids	 = delex_act_ids	
	self.bleu_ref		 = bleu_ref		
	self.bleu_ref_dacts  = bleu_ref_dacts 
	
end

function get_act_slot(dact)
	local dialog_act = dact['act']
	local slots = dact['slots']

	local dacts_sv = {}
	local slots_present = {}
	local values_present = {}

	local dact_sv_string = dialog_act .. '('

	if #slots == 1 then -- 1 arg
		if slots[1][1] == 'slot' then
			table.insert(slots_present,slots[1][2] )
			table.insert(values_present,nil)
			dacts_sv[slots[1][2]] = 'no_value'
			dact_sv_string = dact_sv_string .. slots[1][2] .. ';'
		else
			table.insert(slots_present,slots[1][1])
			table.insert(values_present,slots[1][2])
			dacts_sv[slots[1][1]] = slots[1][2]
			dact_sv_string = dact_sv_string .. slots[1][1] .. '=' .. slots[1][2] .. ';'
		end
	elseif #slots >= 2 then --This happens only for dact 'canthelp'
		for i = 1,#slots do
			table.insert(slots_present,slots[i][1])
			table.insert(values_present,slots[i][2])
			dacts_sv[slots[i][1]] = slots[i][2]
			dact_sv_string = dact_sv_string .. slots[i][1] .. '=' .. slots[i][2] .. ';'
		end
	else
	--else -- 0 args
	--	slot = ''
	end
	--if #slots == 2 then
	--	print('2 slots')
	--	print(dact)
	--end
	dact_sv_string = dact_sv_string .. ')'
	return dialog_act, slots_present, values_present, dacts_sv, dact_sv_string
end

function data:get_tensor_system(acts)
	act_tensor = torch.Tensor(self.NSSacts):fill(0)
	local dacts_sv = {}
	local dacts_sv_string = ''
	local act_table = {}
	for single_act = 1,#acts do
		local act,slots,values,dacts_sv_peract,dact_sv_string = get_act_slot(acts[single_act])
		dacts_sv_string = dacts_sv_string .. dact_sv_string .. ' '
		--print('he')
		--print(dacts_sv_peract)
		--print(dacts_sv_peract)
		for s,v in pairs(dacts_sv_peract) do
			if v ~= 'dontcare' and v ~= 'no_value' then
				if dacts_sv[s] == nil then
					dacts_sv[s] = {tostring(v)}
				else
					table.insert(dacts_sv[s],tostring(v))
					--print(dacts_sv[s])
				end
			end
		end
		if #slots == 0 then
			--print(acts[single_act])
			--print(act)
			--print(self.dialog_ptr[1])
			act_tensor[self.system_actslot_inds[act]] = 1
			table.insert(act_table,self.system_actslot_inds[act])
		elseif #slots == 1 then
			--print(act .. slots[1])
			--print(self.system_actslot_inds[act .. slots[1]])
			--if act_tensor[self.system_actslot_inds[act .. slots[1]]] == 1 then
			--	print(acts)
			--end
			if #values > 0 and values[1] == 'dontcare' then
				slots[1] = slots[1] .. '_dontcare'
			end
			--print(act .. slots[1])
			act_tensor[self.system_actslot_inds[act .. slots[1]]] = 1
			table.insert(act_table,self.system_actslot_inds[act .. slots[1]])
		elseif #slots >= 2 then -- assume it is 'canthelp'
			for i = 1,#slots do
				if act_tensor[self.system_actslot_inds[act .. slots[i]]] == 1 then
					print(acts)
				end
				act_tensor[self.system_actslot_inds[act .. slots[i]]] = 1
				table.insert(act_table,self.system_actslot_inds[act .. slots[i]])
			end
		end
	end
	return act_tensor, dacts_sv, dacts_sv_string, act_table
end

function data:bleu_ref_write()
	file = io.open('../ref_utterances/bleu_ref','w')
	for dact_rep,utts in pairs(self.bleu_ref) do
		file:write(dact_rep,'\n')
		for _,utt in pairs(utts) do file:write(utt,'\n') end
		file:write('\n')
	end
	file:close()

	file = io.open('../ref_utterances/bleu_ref_dacts','w')
	for dact_rep,utts in pairs(self.bleu_ref_dacts) do
		file:write(dact_rep,'\n')
		for _,utt in pairs(utts) do file:write(utt,'\n') end
		file:write('\n')
	end
	file:close()
end

function data:writeStats()
	file = io.open('../ref_utterances/dacts','w')
	for i = 1,self.Ndata do file:write(self.dact[i],'\n') end	
	file:close()

	file = io.open('../ref_utterances/slots','w')
	for i = 1,self.Ndata do
		for dact_rep,utts in pairs(self.dacts_sv[i]) do file:write(dact_rep,',') end
		file:write('\n')
	end
	file:close()
end

function data:writeDataToFile(inds,filename)
	file = io.open(filename,'w')
	io.output(file)
	local i
	for i = 1,#inds do
		io.write(self.dact[inds[i]],'\n')
		io.write(self:features_to_dactstring(self.features[inds[i]]),'\n')
		io.write(self.raw_utt[inds[i]],'\n\n')
	end
	io.close(file)
	io.output(io.stdout)
end


function read_json(fname)
  local f = io.open(fname, "r")
  local content = f:read("*all")
  f:close()
  local luatable = JSON:decode(content)
  return luatable
end
	
function read_file(filename)
	local file = io.open(filename, "r");
	local arr = {}
	for line in file:lines() do
		table.insert (arr, line);
	end
	return arr
end

function read_json(fname)
  local f = io.open(fname, "r")
  local content = f:read("*all")
  f:close()
  local luatable = JSON:decode(content)
  return luatable
end

