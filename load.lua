require('nn')
require('sampler')

local snap = Word2VecSampler.load("sampler.t7")
local sampler = Word2VecSampler(snap)

print(sampler:similar("and", 2))
