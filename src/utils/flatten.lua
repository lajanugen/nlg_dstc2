function flatten(parameters)

   -- returns true if tensor occupies a contiguous region of memory (no holes)
   local function isCompact(tensor)
      local sortedStride, perm = torch.sort(
            torch.LongTensor(tensor:nDimension()):set(tensor:stride()), 1, true)
      local sortedSize = torch.LongTensor(tensor:nDimension()):set(
            tensor:size()):index(1, perm)
      local nRealDim = torch.clamp(sortedStride, 0, 1):sum()
      sortedStride = sortedStride:narrow(1, 1, nRealDim):clone()
      sortedSize   = sortedSize:narrow(1, 1, nRealDim):clone()
      local t = tensor.new():set(tensor:storage(), 1,
                                 sortedSize:storage(),
                                 sortedStride:storage())
      return t:isContiguous()
   end

   if not parameters or #parameters == 0 then
      return torch.Tensor()
   end
   local Tensor = parameters[1].new
   local TmpTensor = Tensor

   -- 1. construct the set of all unique storages referenced by parameter tensors
   local storages = {}
   local nParameters = 0
   local parameterMeta = {}
   for k = 1,#parameters do
      local param = parameters[k]
      local storage = parameters[k]:storage()
      local storageKey = torch.pointer(storage)

      if not storages[storageKey] then
         storages[storageKey] = {storage, nParameters}
         nParameters = nParameters + storage:size()
      end

      parameterMeta[k] = {storageOffset = param:storageOffset() +
                                          storages[storageKey][2],
                          size          = param:size(),
                          stride        = param:stride()}
   end

   -- 2. construct a single tensor that will hold all the parameters
   local flatParameters = TmpTensor(nParameters):zero()

   -- 3. determine if there are elements in the storage that none of the
   --    parameter tensors reference ('holes')
   local tensorsCompact = true
   for k = 1,#parameters do
      local meta = parameterMeta[k]
      local tmp = TmpTensor():set(
         flatParameters:storage(), meta.storageOffset, meta.size, meta.stride)
      tmp:fill(1)
      tensorsCompact = tensorsCompact and isCompact(tmp)
   end

   local maskParameters  = flatParameters:byte():clone()
   local compactOffsets  = flatParameters:long():cumsum(1)
   local nUsedParameters = compactOffsets[-1]

   -- 4. copy storages into the flattened parameter tensor
   for _, storageAndOffset in pairs(storages) do
      local storage, offset = table.unpack(storageAndOffset)
      flatParameters[{{offset+1,offset+storage:size()}}]:copy(Tensor():set(storage))
   end

   -- 5. allow garbage collection
   storages = nil
   for k = 1,#parameters do
       parameters[k]:set(Tensor())
   end

   -- 6. compact the flattened parameters if there were holes
   if nUsedParameters ~= nParameters then
      assert(tensorsCompact,
         "Cannot gather tensors that are not compact")

      flatParameters = TmpTensor(nUsedParameters):copy(
            flatParameters:maskedSelect(maskParameters))
      for k = 1,#parameters do
        parameterMeta[k].storageOffset =
              compactOffsets[parameterMeta[k].storageOffset]
      end
   end

   if TmpTensor ~= Tensor then
      flatParameters = Tensor(flatParameters:nElement()):copy(flatParameters)
   end

   -- 7. fix up the parameter tensors to point at the flattened parameters
   for k = 1,#parameters do
      parameters[k]:set(flatParameters:storage(),
          parameterMeta[k].storageOffset,
          parameterMeta[k].size,
          parameterMeta[k].stride)
   end

   return flatParameters
end

return {flatten=flatten}

