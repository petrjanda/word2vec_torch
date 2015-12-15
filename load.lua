require('sampler')

local snap = Word2VecSampler.load("sampler.t7")
local sampler = Word2VecSampler(snap)
local similar = sampler:similar("nasa", 3)

print(sampler.word2index)
print(similar)
