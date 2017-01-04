function lstm(x, prev_c, prev_h, params)
  -- Calculate all four gates in one go
  local i2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(x)
  local h2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(prev_h)
  local gates = nn.CAddTable()({i2h, h2h})
  
  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slize the n_gates dimension, i.e dimension 2
  local reshaped_gates =  nn.Reshape(4,params.rnn_size)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)
  
  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
  local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
  })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  return next_c, next_h
end

function lstm_cond(x, prev_c, prev_h, prev_d, params)
  -- Calculate all four gates in one go
  local i2h = nn.Linear(params.input_size, 4*params.rnn_size)(x)
  local h2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(prev_h)
  local gates = nn.CAddTable()({i2h, h2h})
  
  local i2d = nn.Linear(params.input_size, params.cond_len)(x)
  local h2d = nn.Linear(params.rnn_size, params.cond_len)(prev_h)
  local read_gate = nn.Sigmoid()(nn.CAddTable()({i2d, h2d}))
  -- read_gate = nn.Reshape(params.cond_len,params.batch_size)(read_gate)
  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slize the n_gates dimension, i.e dimension 2
  local reshaped_gates =  nn.Reshape(4,params.rnn_size)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)
  
  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
  local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

  local next_d = nn.CMulTable()({prev_d,read_gate}) 
  local lin_proj = nn.Linear(params.cond_len, params.rnn_size)(next_d)
  local d_to_c = nn.Tanh()(lin_proj)

  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform}), 
	  d_to_c
  })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  --next_d = nn.Transpose({1,2})(next_d)

  return next_c, next_h, next_d
end

function read_gate_upd(x, prev_h_table, params, prev_d)
  local i2d = nn.Linear(params.input_size, params.cond_len)(x)
  local read_input = i2d
  local i, prev_h
  for i = 1,params.layers do
	local prev_h = prev_h_table[2*i]
	local alpha  = params.alphas[i]
	local h2d = nn.Linear(params.rnn_size, params.cond_len)(prev_h)
	h2d = nn.MulConstant(alpha)(h2d)
	read_input = nn.CAddTable()({read_input, h2d})
  end
  local read_gate = nn.Sigmoid()(read_input)
  local next_d = nn.CMulTable()({prev_d,read_gate}) 
  --return read_gate, next_d
  return next_d
end

function lstm_orig_2inp(x, prev_c, prev_h, params, next_d)
  -- Calculate all four gates in one go
  local i2h = nn.Linear(params.input_size, 4*params.rnn_size)(x)
  local h2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(prev_h)
  local gates = nn.CAddTable()({i2h, h2h})
  
  -- Then slize the n_gates dimension, i.e dimension 2
  local reshaped_gates =  nn.Reshape(4,params.rnn_size)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)
  
  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
  local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

  local lin_proj = nn.Linear(params.cond_len, params.rnn_size)(next_d)
  local d_to_c = nn.Tanh()(lin_proj)

  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform}), 
	  d_to_c
  })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  --next_d = nn.Transpose({1,2})(next_d)

  return next_c, next_h
end

function lstm_orig_3inp(x0, x1, prev_c, prev_h, params, next_d)
  -- Calculate all four gates in one go
  local i2h0 = nn.Linear(params.input_size, 4*params.rnn_size)(x0)
  local i2h1 = nn.Linear(params.input_size, 4*params.rnn_size)(x1)
  local h2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(prev_h)
  local gates = nn.CAddTable()({i2h0, i2h1, h2h})
  
  -- Then slize the n_gates dimension, i.e dimension 2
  local reshaped_gates =  nn.Reshape(4,params.rnn_size)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)
  
  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
  local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

  local lin_proj = nn.Linear(params.cond_len, params.rnn_size)(next_d)
  local d_to_c = nn.Tanh()(lin_proj)

  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform}), 
	  d_to_c
  })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  --next_d = nn.Transpose({1,2})(next_d)

  return next_c, next_h
end

