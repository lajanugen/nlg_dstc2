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
require('online_sampler')
require('data')

local NLG = torch.class('NLG')
function NLG:__init()--(save_path,input_size,max_epoch)
	g_init_gpu(arg)

	print('Loading Data properties')
	self.data_props	= torch.load('../pre-trained/data_properties')
	print('Loading model')
	params			= torch.load('../pre-trained/run_params')
	model_params	= torch.load('../pre-trained/params')
	sampling_params	= torch.load('../pre-trained/sampling_params')
	print('Initializing Sampler')
	self.sampler = sampler(params,sampling_params,model_params,nil,self.data_props)
	print('Ready for input')
end


function NLG:get_tensor_system(acts)
	local act_tensor = torch.Tensor(self.data_props.NSSacts,1):fill(0)
	local dacts_sv = {}
	local dacts_sv_string = ''
	local sv = {}
	for single_act = 1,#acts['dasv'] do
		local dialog_act = acts['dasv'][single_act]
		local act = dialog_act['dialog-act']
		local slot = dialog_act['slot']
		local value = dialog_act['value']


		if (slot == '') or (slot == nil) then
			act_tensor[self.data_props.system_actslot_inds[act]] = 1
		else
			--if value == '' then
			if value == 'dontcare' then
				slot = slot .. '_dontcare'
			elseif value ~= 'no_value' then
				sv[slot] = value
			end
			act_tensor[self.data_props.system_actslot_inds[act .. slot]] = 1
		end
	end
	act_tensor = transfer_data(act_tensor)
	return act_tensor, sv
end

function pretty_print_input(json)
	local total_dact = json['dasv']
	local dact_sv_string = ''
	for i = 1,#total_dact do
		local dact = total_dact[i]
		local act = dact['dialog-act']
		local slot = dact['slot']
		local value = dact['value']
		if slot ~= nil and slot ~= '' then
			if value ~= nil and value ~= '' then
				dact_sv_string = dact_sv_string .. act .. '(' .. slot .. '=' .. value .. ') '
			else
				dact_sv_string = dact_sv_string .. act .. '(' .. slot .. ') '
			end
		else
			dact_sv_string = dact_sv_string .. act .. '() '
		end
	end
	return dact_sv_string
end

function NLG:serve(json_input)

	--local size = 2^13      -- good buffer size (8K)
	--local input = io.read(size)

	--input_table = JSON:decode(input)
	
	input_table = read_json(json_input)
	
	local sys, sv = self:get_tensor_system(input_table)
	--n = 1
	--m = 1
	local sample = self.sampler:sample(sys,sv)

	--io.write(sample)
	--print(sample)
	local dact_sv_string = pretty_print_input(input_table)
	print('System action:\t',dact_sv_string)
	print('Generated Sentence:',sample)
	--io.write(JSON:encode({text=sample,["nlg-version"]=1}))
	--return JSON:encode({text=sample,["nlg-version"]=1})

end
