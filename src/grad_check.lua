local function grad_check()--(save_path,input_size,max_epoch)

	print("Network parameters:")
	params.batch_size = 1
	local states = {state_train, state_valid, state_test}
	for _, state in pairs(states) do
	  reset_state(state)
	end
	setup()
	local step = 0
	local epoch = 0
	local total_cases = 0
	local beginning_time = torch.tic()
	local start_time = torch.tic()
	print("Starting training.")
	local words_per_step = params.seq_length * params.batch_size

	local epoch_size = state_train.batch_count
	
	local train_errors = {}
	local valid_errors = {}
	local train_bleus = {}
	local valid_bleus = {}

	state_train.feats:fill(1)

	local err1 = fp(state_train)
	state_train.sent = 2
	bp(state_train)

	local t
	for t = 1, paramx:size(1) do

		state_train.pos = 1
		local param_value = paramx:sub(t,t)[1]
		paramx:sub(t,t):fill(param_value + params.grad_check_eps)
		local err2 = fp(state_train)
		paramx:sub(t,t):fill(param_value)
		--print('param',paramx:sub(1,1))

		local numerical_grad = (err2 - err1) / params.grad_check_eps

		state_train.pos = 1
		--fp(state_train)
		--state_train.sent = 2
		--bp(state_train)
		--print(numerical_grad)
		--if math.abs(numerical_grad - paramdx:sub(t,t)[1]) > 0.001 then
			print(numerical_grad,paramdx:sub(t,t)[1])
		--end
	end
end


