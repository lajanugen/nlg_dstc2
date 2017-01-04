function create_network()
  local x                = nn.Identity()()
  local y                = nn.Identity()()
  local prev_s           = nn.Identity()()
  -- i[0] = embedding vector of x
  -- i is used to store the output h of the lstm after each time step
  local i
    i                = {[0] = LookupTable(params.vocab_size,params.rnn_size)(x)}
  local next_s           = {}
  local split         = {prev_s:split(2 * params.layers)}
  for layer_idx = 1, params.layers do
    local prev_c         = split[2 * layer_idx - 1]
    local prev_h         = split[2 * layer_idx]
    local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
    local next_c, next_h = lstm(dropped, prev_c, prev_h, params)
    table.insert(next_s, next_c)
    table.insert(nexr_s, next_h)
    i[layer_idx] = next_h
  end
  local h2y              = nn.Linear(params.rnn_size, params.vocab_size)
  local dropped          = nn.Dropout(params.dropout)(i[params.layers])
  local pred             = nn.LogSoftMax()(h2y(dropped))
  local err              = nn.ClassNLLCriterion()({pred, y})
  local module           = nn.gModule({x, y, prev_s},
                                      {err, nn.Identity()(next_s)})
  module:getParameters():uniform(-params.init_weight, params.init_weight)
  return transfer_data(module)
end

function create_network_orig3()
  local x                = nn.Identity()()
  local y                = nn.Identity()()
  local prev_s           = nn.Identity()()
  local prev_d           = nn.Identity()()
  -- i[0] = embedding vector of x
  -- i is used to store the output h of the lstm after each time step
  if params.pretrained == 1 then
	local embedding = GloVeEmbeddingFixed(vocab_map, 300, '')--, 'restaurant_vectors')
  	i = {[0] = embedding(x)}
  elseif params.pretrained == 2 then
	local embedding = GloVeEmbedding(vocab_map, 300, '')--, 'restaurant_vectors')
  	i = {[0] = embedding(x)}
  else
    --i                = {[0] = LookupTable(params.vocab_size,params.input_size)(x)}
    i                = {[0] = nn.Linear(params.vocab_size,params.rnn_size)(x)}
  end

  --local i                = {[0] = LookupTable(params.vocab_size,
  --                                                  params.input_size)(x)}
  local next_s           = {}
  local next_d	         = {}
  local split			 = {prev_s:split(2 * params.layers)}

  --local read_gate, next_d = read_gate_upd(i[0], split, params, prev_d)
  local next_d = read_gate_upd(i[0], split, params, prev_d)

  for layer_idx = 1, params.layers do
    local prev_c         = split[2 * layer_idx - 1]
    local prev_h         = split[2 * layer_idx]
    local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
	local next_c, next_h
	if layer_idx == 1 then
		next_c, next_h		 = lstm_orig_2inp(dropped, prev_c, prev_h, params, next_d)
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

  local pred       = nn.LogSoftMax()(softmax_in) -- batch_size x vocab_size
  --local cross_ent  = nn.ClassNLLCriterion()({pred, y})
  local logprobs = nn.CMulTable()({y, pred})
  local logprobs = nn.Sum(2)(logprobs)
  local logprobs = nn.MulConstant(-1)(logprobs)

  local dist_d			 = nn.PairwiseDistance(2)({next_d,prev_d})
  local dist_exp		 = nn.Exp()(dist_d)
  local dist_pow		 = nn.Power(math.log(params.xi))(dist_exp) --exp(xlna) = a^x
  local dist_pen		 = nn.MulConstant(params.eta)(dist_pow)
  --local dist_pen		 = nn.MulConstant(1.0/params.batch_size)(dist_pen)
  --local err				 = nn.CAddTable()({cross_ent, nn.MulConstant(1.0/params.batch_size)(nn.Sum()(dist_pen))})
  local err				 = nn.CAddTable()({logprobs, dist_pen})

  --local err = logprobs

  local y_mask = nn.Sum(2)(y)

  local err = nn.DotProduct()({err, y_mask})

  local err = nn.MulConstant(1.0/params.batch_size)(err)

  --local err = nn.Sum()(err)

  if not params.obj_exp_term then
	err = cross_ent
  end
  
  local module           = nn.gModule({x, y, prev_s, prev_d},
                                      {err, nn.Identity()(next_s), nn.Identity()(next_d)})
  module:getParameters():uniform(-params.init_weight, params.init_weight)
  return transfer_data(module)
end

function create_network_orig2()
  local x                = nn.Identity()()
  local y                = nn.Identity()()
  local prev_s           = nn.Identity()()
  local prev_d           = nn.Identity()()
  -- i[0] = embedding vector of x
  -- i is used to store the output h of the lstm after each time step
  if params.pretrained == 1 then
	local embedding = GloVeEmbeddingFixed(vocab_map, 300, '')--, 'restaurant_vectors')
  	i = {[0] = embedding(x)}
  elseif params.pretrained == 2 then
	local embedding = GloVeEmbedding(vocab_map, 300, '')--, 'restaurant_vectors')
  	i = {[0] = embedding(x)}
  else
    i                = {[0] = LookupTable(params.vocab_size,params.input_size)(x)}
  end

  --local i                = {[0] = LookupTable(params.vocab_size,
  --                                                  params.input_size)(x)}
  local next_s           = {}
  local next_d	         = {}
  local split			 = {prev_s:split(2 * params.layers)}

  --local read_gate, next_d = read_gate_upd(i[0], split, params, prev_d)
  local next_d = read_gate_upd(i[0], split, params, prev_d)

  for layer_idx = 1, params.layers do
    local prev_c         = split[2 * layer_idx - 1]
    local prev_h         = split[2 * layer_idx]
    local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
	local next_c, next_h
	if layer_idx == 1 then
		next_c, next_h		 = lstm_orig_2inp(dropped, prev_c, prev_h, params, next_d)
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

  local pred       = nn.LogSoftMax()(softmax_in) -- batch_size x vocab_size
  --local cross_ent  = nn.ClassNLLCriterion()({pred, y})
  local logprobs = nn.CMulTable()({y, pred})
  local logprobs = nn.Sum(2)(logprobs)
  local logprobs = nn.MulConstant(-1)(logprobs)

  local dist_d			 = nn.PairwiseDistance(2)({next_d,prev_d})
  local dist_exp		 = nn.Exp()(dist_d)
  local dist_pow		 = nn.Power(math.log(params.xi))(dist_exp) --exp(xlna) = a^x
  local dist_pen		 = nn.MulConstant(params.eta)(dist_pow)
  --local dist_pen		 = nn.MulConstant(1.0/params.batch_size)(dist_pen)
  --local err				 = nn.CAddTable()({cross_ent, nn.MulConstant(1.0/params.batch_size)(nn.Sum()(dist_pen))})
  local err				 = nn.CAddTable()({logprobs, dist_pen})

  --local err = logprobs

  local y_mask = nn.Sum(2)(y)

  local err = nn.DotProduct()({err, y_mask})

  local err = nn.MulConstant(1.0/params.batch_size)(err)

  --local err = nn.Sum()(err)

  if not params.obj_exp_term then
	err = cross_ent
  end
  
  local module           = nn.gModule({x, y, prev_s, prev_d},
                                      {err, nn.Identity()(next_s), nn.Identity()(next_d)})
  module:getParameters():uniform(-params.init_weight, params.init_weight)
  return transfer_data(module)
end


function create_network_orig()
  local x                = nn.Identity()()
  local y                = nn.Identity()()
  local prev_s           = nn.Identity()()
  local prev_d           = nn.Identity()()
  -- i[0] = embedding vector of x
  -- i is used to store the output h of the lstm after each time step
  if params.pretrained == 1 then
	local embedding = GloVeEmbeddingFixed(vocab_map, 300, '')--, 'restaurant_vectors')
  	i = {[0] = embedding(x)}
  elseif params.pretrained == 2 then
	local embedding = GloVeEmbedding(vocab_map, 300, '')--, 'restaurant_vectors')
  	i = {[0] = embedding(x)}
  else
    i                = {[0] = LookupTable(params.vocab_size,params.input_size)(x)}
  end

  --local i                = {[0] = LookupTable(params.vocab_size,
  --                                                  params.input_size)(x)}
  local next_s           = {}
  local next_d	         = {}
  local split			 = {prev_s:split(2 * params.layers)}

  --local read_gate, next_d = read_gate_upd(i[0], split, params, prev_d)
  local next_d = read_gate_upd(i[0], split, params, prev_d)

  for layer_idx = 1, params.layers do
    local prev_c         = split[2 * layer_idx - 1]
    local prev_h         = split[2 * layer_idx]
    local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
	local next_c, next_h
	if layer_idx == 1 then
		next_c, next_h		 = lstm_orig_2inp(dropped, prev_c, prev_h, params, next_d)
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

  local pred       = nn.LogSoftMax()(softmax_in)
  local cross_ent  = nn.ClassNLLCriterion()({pred, y})

  local dist_d			 = nn.PairwiseDistance(2)({next_d,prev_d})
  local dist_exp		 = nn.Exp()(dist_d)
  local dist_pow		 = nn.Power(math.log(params.xi))(dist_exp) --exp(xlna) = a^x
  local dist_pen		 = nn.MulConstant(params.eta)(dist_pow)
  local err				 = nn.CAddTable()({cross_ent, nn.MulConstant(1.0/params.batch_size)(nn.Sum()(dist_pen))})
  --local err				 = nn.CAddTable()({cross_ent, nn.Sum()(dist_pen)})
  if not params.obj_exp_term then
	err = cross_ent
  end
  
  local module           = nn.gModule({x, y, prev_s, prev_d},
                                      {err, nn.Identity()(next_s), nn.Identity()(next_d)})
  module:getParameters():uniform(-params.init_weight, params.init_weight)
  return transfer_data(module)
end

function create_network_cond()
  local x                = nn.Identity()()
  local y                = nn.Identity()()
  local prev_s           = nn.Identity()()
  local prev_d           = nn.Identity()()
  -- i[0] = embedding vector of x
  -- i is used to store the output h of the lstm after each time step
  if params.pretrained == 1 then
	local embedding = GloVeEmbeddingFixed(vocab_map, 300, '')--, 'restaurant_vectors')
  	i = {[0] = embedding(x)}
  elseif params.pretrained == 2 then
	local embedding = GloVeEmbedding(vocab_map, 300, '')--, 'restaurant_vectors')
  	i = {[0] = embedding(x)}
  else
    i                = {[0] = LookupTable(params.vocab_size,params.input_size)(x)}
  end

  --local i                = {[0] = LookupTable(params.vocab_size,
  --                                                  params.input_size)(x)}
  local next_s           = {}
  local next_d	         = {}
  local split         = {prev_s:split(2 * params.layers)}
  for layer_idx = 1, params.layers do
    local prev_c         = split[2 * layer_idx - 1]
    local prev_h         = split[2 * layer_idx]
    local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
	local next_c, next_h
	if layer_idx == 1 then
		next_c, next_h, next_d = lstm_cond(dropped, prev_c, prev_h, prev_d, params)
		--local next_c, next_h = lstm(dropped, prev_c, prev_h)
	else
		next_c, next_h = lstm(dropped, prev_c, prev_h, params)
	end
	table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end
  local h2y              = nn.Linear(params.rnn_size, params.vocab_size)
  local dropped          = nn.Dropout(params.dropout)(i[params.layers])
  local pred             = nn.LogSoftMax()(h2y(dropped))
  local cross_ent		 = nn.ClassNLLCriterion()({pred, y})

  --local dist_d			 = nn.PairwiseDistance(2)({nn.Transpose({1,2})(next_d),nn.Transpose({1,2})(prev_d)})
  local dist_d			 = nn.PairwiseDistance(2)({next_d,prev_d})
  local dist_exp		 = nn.Exp()(dist_d)
  local dist_pow		 = nn.Power(math.log(params.xi))(dist_exp) --exp(xlna) = a^x
  local dist_pen		 = nn.MulConstant(params.eta)(dist_pow)
  local err				 = nn.CAddTable()({cross_ent, nn.MulConstant(1.0/params.batch_size)(nn.Sum()(dist_pen))})
  --local err = cross_ent
  
  local module           = nn.gModule({x, y, prev_s, prev_d},
                                      {err, nn.Identity()(next_s), nn.Identity()(next_d)})
  module:getParameters():uniform(-params.init_weight, params.init_weight)
  return transfer_data(module)
end


