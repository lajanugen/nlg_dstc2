
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

function data:read_data()
	local data_json = read_json("../../restaurant_data/sfxrestaurant/train+valid+test.json")
	self.dact = {}
	self.raw_utt = {}
	local dact_sent = {}
	for i = 1,#data_json do
		local dialog = data_json[i]['dial']
		for j = 1,#dialog do 
			local dact = dialog[j]['S']['dact']
			local sent = dialog[j]['S']['ref']
			if self.unique_instances then
				if not (dact_sent[dact] and dact_sent[dact][sent]) then
					table.insert(self.dact,dact)
					table.insert(self.raw_utt,sent)
				end
			else
				table.insert(self.dact,dact)
				table.insert(self.raw_utt,sent)
			end
			if dact_sent[dact] == nil then 
				dact_sent[dact] = {}
			end
			dact_sent[dact][sent] = true
		end
	end
	self.dact_sent = dact_sent
	self.Ndata = #self.dact
	print(self.Ndata)
end

function data:parse_data()
	self.acts = {}
	self.slots = {}
	self.values = {}
	self.dacts_sv = {}
	self.dacts = {}

	for i = 1,self.Ndata do
	    local act = self.dact[i]
	    local s = string.find(act,'[(]')
	    local e = string.find(act,'[)]')
	    local dialog_act = string.sub(act,1,s-1)
	    self.acts[dialog_act] = true
	    self.dacts[i] = dialog_act
	    self.dacts_sv[i] = {}

		if s + 1 < e then slots_values = string.sub(act,s+1,e-1) 
		else slots_values = '' 
		end

		slots_values = string.split(slots_values,';')

		for _,slot_value in ipairs(slots_values) do
			local slot_value_tab = string.split(slot_value,'=')
			local slot = slot_value_tab[1]
			local value
			if #slot_value_tab > 1 then value = slot_value_tab[2]
			else						value = '1'
			end
			if (string.sub(value,1,1) == "'") and (string.sub(value,-1,-1) == "'") then value = string.sub(value,2,-2) end
			self.slots[slot] = true
			if self.values[slot] == nil then self.values[slot] = {} end
			self.values[slot][value] = true
			--if dacts_sv[i][slot] ~= nil then print(act) end
			--assert(dacts_sv[i][slot] == nil, 'Duplicate values')
			self.dacts_sv[i][slot] = value
		end
	end
end	

function data:get_mappings()

	local dact_number = {}
	local idact_number = {}
	local slot_number = {}
	local islot_number = {}
	
	local ct = 1
	for slot,_ in pairs(self.slots) do
		slot_number[slot] = ct
		islot_number[ct] = slot
		ct = ct + 1
	end
	self.Nslots = ct - 1

	ct = 1
	for act,_ in pairs(self.acts) do
		dact_number[act] = ct
		idact_number[ct] = act
		ct = ct + 1
	end
	self.Nacts = ct - 1

	local slot_ind = 0
	local slot_index = {}
	local islot_index = {}
	for slot,_ in pairs(self.slots) do
		slot_index[slot] = 'value'..tostring(slot_ind)
		islot_index[slot_index[slot]] = slot
		slot_ind = slot_ind + 1
	end

	local dact_ind = 1
	local dact_index = {}
	local idact_index = {}
	for dact,_ in pairs(self.acts) do
		dact_index[dact] = dact_ind
		idact_index[dact_index[dact]] = dact
		dact_ind = dact_ind + 1
	end

	self.dact_number	= dact_number
	self.idact_number	= idact_number	
	self.slot_number	= slot_number	
	self.islot_number	= islot_number	
	self.slot_index		= slot_index	
	self.islot_index	= islot_index	
	self.dact_index		= dact_index	
	self.idact_index	= idact_index	

end

function data:delexicalize()
	local raw_utt = self.raw_utt
	local dacts_sv = self.dacts_sv
	local slot_index = self.slot_index

	local raw_utt_delex = {}
	local raw_utt_token = {}
	local lengths = {}
	for i = 1,#dacts_sv do
		utterance = raw_utt[i]
		for slot,_ in pairs(dacts_sv[i]) do
			value = dacts_sv[i][slot]
			if value ~= nil then
				if (string.sub(value,1,1) == "'") and (string.sub(value,-1,-1) == "'") then
					value = string.sub(value,2,-2)
				end
				u,v = string.find(utterance, value)
				if (u~=nil) then
					before = string.sub(utterance,u-1,u-1)
					after = string.sub(utterance,v+1,v+1)
					if not(string.match(before,'%w') or string.match(after,'%w')) then
						utterance = string.sub(utterance,1,u-1) .. slot_index[slot] .. string.sub(utterance,v+1)
					end
				end
			end
		end
		table.insert(raw_utt_delex, utterance)
		raw_utt_token[i] = string.split(utterance," ")
		lengths[i] = #raw_utt_token[i]
	end	
	self.raw_utt_delex	= raw_utt_delex --[i] = delexicalized utterance
	self.raw_utt_token	= raw_utt_token --[i] = delexicalized utterance tokens
	self.lengths		= lengths		--[i] = lenths of delexicalized utterances
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
	--for i = 1, 10 do
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

	for i = 1,self.Nacts do
		vocab_idx = vocab_idx + 1
		vocab_map['dact_' .. self.idact_index[i]] = vocab_idx
		ivocab_map[vocab_idx] = 'dact_' .. self.idact_index[i]
	end

	vocab_idx = vocab_idx + 1
	ivocab_map[vocab_idx] = 'presence of slot'
	vocab_map['presence of slot'] = vocab_idx

	vocab_idx = vocab_idx + 1
	ivocab_map[vocab_idx] = 'absence of slot'
	vocab_map['absence of slot'] = vocab_idx

	vocab_idx = vocab_idx + 1
	ivocab_map[vocab_idx] = 'dont_care'
	vocab_map['dont_care'] = vocab_idx

	--return vocab_map, ivocab_map, x
	self.vocab_map = vocab_map   --word -> idx
	self.ivocab_map = ivocab_map --idx  -> word
	self.x = x					 --[i]  -> '1' + word indices + 'w'
	self.vocab_size = vocab_idx
end	

function data:determine_split()

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

	local train_inds,train_inds_uniq,train_inds_uniq_marker = {},{},{}
	local valid_inds,valid_inds_uniq,valid_inds_uniq_marker = {},{},{}
	local test_inds,test_inds_uniq,test_inds_uniq_marker	= {},{},{}
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

function data:organize_mask(seq,inds,size)

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

	for i = 1,size do
		sentence_id = {}
		local index = inds[i]

		for ct = 1,#seq[index] do
			local word = seq[index][ct]
			batch_senoh[ct][i][word] = 1
			batch_mask[ct][i] = 1
		end
		batch_mask[#seq[index]][i] = 0 --This is so that the last instance where mask is 1 is when the output is STOP_SYMBOL
		table.insert(sentence_id,index)
	end

	return {transfer_data(batch_senoh), transfer_data(batch_mask), sentence_id}
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
	local source_seq = self:organize_mask(self.features,inds,sz)
	local target_seq = self:organize_mask(x,inds,sz)
	return {source_seq[1], source_seq[2], target_seq[1], target_seq[2], source_seq[3]}
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
	local source_seq = self:organize_mask(self.features,inds,self.batch_size)
	local target_seq = self:organize_mask(x,inds,self.batch_size)
	return {source_seq[1], source_seq[2], target_seq[1], target_seq[2], source_seq[3]}
end

function data:dact_features()
	local feat_len = self.Nacts + self.Nslots
	local features = torch.zeros(self.Ndata,feat_len)

	for i = 1, self.Ndata do
		if self.dact_number[self.dacts[i]] == nil then return nil end
		features[i][self.dact_number[self.dacts[i]]] = 1
		local slot, value
		for slot,value in pairs(self.dacts_sv[i]) do
			if self.params.dont_care_handle and (value == 'dont_care') then
				features[i][self.Nacts + self.slot_number[slot..'_'..value]] = 1
			end
			assert(self.slot_number[slot] ~= nil, "nil slot")
			features[i][self.Nacts + self.slot_number[slot]] = 1
		end
	end

	self.features = features
	self.feat_len = feat_len
	return features, feat_len
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
	    table.insert(dact_rep,self.dact_number[self.dacts[i]])
		for slot,value in pairs(self.dacts_sv[i]) do
			if (value == 'dont_care') then token = slot ..'_'..value end
			if care_values then table.insert(dact_rep,self.slot_number[token])
			else				table.insert(dact_rep,self.slot_number[slot]) end
		end
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
		io.write(self.dact[inds[i]])
		io.write('\n')
		io.write(self.raw_utt[inds[i]])
		io.write('\n\n')
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
	
