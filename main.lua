--[[
Config file for skipgram with negative sampling 
--]]

require("io")
require("os")
require("paths")
require("torch")
dofile("word2vec.lua")
require("sampler.lua")

-- Default configuration
config = {}
config.corpus = "corpus.txt" -- input data
config.window = 5 -- (maximum) window size
config.dim = 100 -- dimensionality of word embeddings
config.alpha = 0.75 -- smooth out unigram frequencies
config.table_size = 1e8 -- table size from which to sample neg samples
config.neg_samples = 5 -- number of negative samples for each positive sample
config.minfreq = 10 --threshold for vocab frequency
config.lr = 0.025 -- initial learning rate
config.min_lr = 0.001 -- min learning rate
config.epochs = 1 -- number of epochs to train
config.gpu = 0 -- 1 = use gpu, 0 = use cpu

-- Parse input arguments
cmd = torch.CmdLine()
cmd:option("-corpus", config.corpus)
cmd:option("-window", config.window)
cmd:option("-minfreq", config.minfreq)
cmd:option("-dim", config.dim)
cmd:option("-lr", config.lr)
cmd:option("-min_lr", config.min_lr)
cmd:option("-neg_samples", config.neg_samples)
cmd:option("-table_size", config.table_size)
cmd:option("-epochs", config.epochs)
cmd:option("-gpu", config.gpu)
params = cmd:parse(arg)

for param, value in pairs(params) do
    config[param] = value
end

-- Print config
for i,j in pairs(config) do
    print(i..": ", j)
end

-- Load corpus 
print("Building vocabulary...")
local start = sys.clock()

local c = Corpus()
c:read(config.corpus)
c:filter(config.minfreq)
c:buildIndices()

print(string.format("%d words and %d sentences processed in %.2f seconds.", c.total, c.lines, sys.clock() - start))
print(string.format("Vocab size after eliminating words occuring less than %d times: %d", config.minfreq, c.vocab_size))

-- Build model
local model = Word2Vec(config, c)

-- Training
for k = 1, config.epochs do
    model.lr = config.lr -- reset learning rate at each epoch
    model:train_stream(config.corpus)
end

-- Create sampler from the model
local sampler = Word2VecSampler{
  word_vecs = model.word_vecs,
  word2index = model.c.word2index,
  index2word = model.c.index2word
}

-- Save sampler to the file
sampler:save("sampler.t7")

print("Done.")
