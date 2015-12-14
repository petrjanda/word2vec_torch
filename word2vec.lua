--[[
  Class for word2vec with skipgram and negative sampling
--]]

require("sys")
require("nn")
require("corpus")

local Word2Vec = torch.class("Word2Vec")

function Word2Vec:__init(config)
  self.gpu = config.gpu -- 1 if train on gpu, otherwise cpu
  self.stream = config.stream -- 1 if stream from hard drive, 0 otherwise
  self.neg_samples = config.neg_samples
  self.minfreq = config.minfreq
  self.dim = config.dim
  self.window = config.window
  self.lr = config.lr
  self.min_lr = config.min_lr
  self.alpha = config.alpha
  self.table_size = config.table_size

  self.tensortype = torch.getdefaulttensortype()
  self.criterion = nn.BCECriterion() -- logistic loss
  self.word = torch.IntTensor(1)
  self.contexts = torch.IntTensor(1 + self.neg_samples) 
  self.labels = torch.zeros(1 + self.neg_samples)
  self.labels[1] = 1 -- first label is always pos sample

  self.c = Corpus()
end

function Word2Vec:save(path)
  local snap = self

  torch.save(path, snap)
end

function Word2Vec.load(path)
  return torch.load(path)
end

-- move to cuda
function Word2Vec:cuda()
  require("cunn")
  require("cutorch")

  cutorch.setDevice(1)
  self.word = self.word:cuda()
  self.contexts = self.contexts:cuda()
  self.labels = self.labels:cuda()
  self.criterion:cuda()
  self.w2v:cuda()
end

-- Build vocab
function Word2Vec:build_vocab(corpus)
  print("Building vocabulary...")
  local start = sys.clock()
  self.c:read(corpus)
  self.c:filter(self.minfreq)
  self.c:buildIndices()

  local vocab_size = self.c.vocab_size 
  local lines = self.c.lines
  local total_count = self.c.total

  print(string.format("%d words and %d sentences processed in %.2f seconds.", total_count, lines, sys.clock() - start))
  print(string.format("Vocab size after eliminating words occuring less than %d times: %d", self.minfreq, vocab_size))

  -- initialize word/context embeddings now that vocab size is known
  self.word_vecs = nn.LookupTable(vocab_size, self.dim) -- word embeddings
  self.context_vecs = nn.LookupTable(vocab_size, self.dim) -- context embeddings
  self.word_vecs:reset(0.25); self.context_vecs:reset(0.25) -- rescale N(0,1)
  self.w2v = nn.Sequential()
  self.w2v:add(nn.ParallelTable())
  self.w2v.modules[1]:add(self.context_vecs)
  self.w2v.modules[1]:add(self.word_vecs)
  self.w2v:add(nn.MM(false, true)) -- dot prod and sigmoid to get probabilities
  self.w2v:add(nn.Sigmoid())

  self.decay = (self.min_lr-self.lr)/(total_count)
end

-- Build a table of unigram frequencies from which to obtain negative samples
function Word2Vec:build_table()
  local start = sys.clock()
  print("Building a table of unigram frequencies... ")
  self.table = self.c:buildUnigramsTable(self.alpha, self.table_size)
  print(string.format("Done in %.2f seconds.", sys.clock() - start))
end

-- Train on word context pairs
function Word2Vec:train_pair(word, contexts)
  local p = self.w2v:forward({contexts, word})
  local loss = self.criterion:forward(p, self.labels)
  local dl_dp = self.criterion:backward(p, self.labels)

  self.w2v:zeroGradParameters()
  self.w2v:backward({contexts, word}, dl_dp)
  self.w2v:updateParameters(self.lr)
end

-- Sample negative contexts
function Word2Vec:sample_contexts(context)
  self.contexts[1] = context
  local i = 0

  while i < self.neg_samples do
    neg_context = self.table[torch.random(self.table_size)]

    if context ~= neg_context then
      self.contexts[i+2] = neg_context
      i = i + 1
    end
  end
end

-- Train on sentences that are streamed from the hard drive
-- Check train_mem function to train from memory (after pre-loading data into tensor)
function Word2Vec:train_stream(corpus)
  print("Training...")
  local start = sys.clock()
  local c = 0

  function process(sentense)
    for i, word in ipairs(sentence) do
      word_idx = self.c:getIndex(word)
      if word_idx ~= nil then -- word exists in vocab
        local reduced_window = torch.random(self.window) -- pick random window size

        self.word[1] = word_idx -- update current word
        self.lr = math.max(self.min_lr, self.lr + self.decay) 

        for j = i - reduced_window, i + reduced_window do -- loop through contexts
          local context = sentence[j]

          if context ~= nil and j ~= i then -- possible context
            context_idx = self.c:getIndex(context)

            if context_idx ~= nil then -- valid context
              self:sample_contexts(context_idx) -- update pos/neg contexts
              self:train_pair(self.word, self.contexts) -- train word context pair

              c = c + 1

              if c % 1000 ==0 then
                print(string.format("%d words trained in %.2f seconds. Learning rate: %.4f", c, sys.clock() - start, self.lr))
              end
            end
          end
        end
      end
    end
  end

  self.c:streamSentenses(corpus, process)
end
