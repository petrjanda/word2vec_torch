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
  self.index2word = {}
  self.word2index = {}
  self.total_count = 0

  self.c = Corpus()
end

function Word2Vec:save(path)
  local snap = {}

  torch.save(path, snap)
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

-- Build vocab frequency, word2index, and index2word from input file
function Word2Vec:build_vocab(corpus)
  print("Building vocabulary...")
  local start = sys.clock()
  self.c:read(corpus)
  self.c:filter(self.minfreq)
  self.c:buildIndices()

  local vocab_size = self.c.vocab_size 

  self.index2word = self.c.index2word
  self.word2index = self.c.word2index
  n = self.c.lines
  self.total_count = self.c.total

  print(string.format("%d words and %d sentences processed in %.2f seconds.", self.total_count, n, sys.clock() - start))
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

  self.decay = (self.min_lr-self.lr)/(self.total_count)

  print("min: ", self.min_lr)
  print("lr: ", self.lr)
  print("total_count: ", self.total_count)
  print("window: ", self.window)
  print("decay: ", self.decay)
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
      word_idx = self.word2index[word]
      if word_idx ~= nil then -- word exists in vocab
        local reduced_window = torch.random(self.window) -- pick random window size
        self.word[1] = word_idx -- update current word

        self.lr = math.max(self.min_lr, self.lr + self.decay) 

        for j = i - reduced_window, i + reduced_window do -- loop through contexts
          local context = sentence[j]
          if context ~= nil and j ~= i then -- possible context
            context_idx = self.word2index[context]
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

-- Row-normalize a matrix
function Word2Vec:normalize(m)
  m_norm = torch.zeros(m:size())
  for i = 1, m:size(1) do
    m_norm[i] = m[i] / torch.norm(m[i])
  end
  return m_norm
end

-- Return the k-nearest words to a word or a vector based on cosine similarity
-- w can be a string such as "king" or a vector for ("king" - "queen" + "man")
function Word2Vec:get_sim_words(w, k)
  if self.word_vecs_norm == nil then
    self.word_vecs_norm = self:normalize(self.word_vecs.weight:double())
  end

  if type(w) == "string" then
    if self.word2index[w] == nil then
      print("'"..w.."' does not exist in vocabulary.")
      return nil
    else
      w = self.word_vecs_norm[self.word2index[w]]
    end
  end

  local sim = torch.mv(self.word_vecs_norm, w)
  sim, idx = torch.sort(-sim)
  local r = {}

  for i = 1, k do
    r[i] = {self.index2word[idx[i]], -sim[i]}
  end

  return r
end

-- print similar words
function Word2Vec:print_sim_words(words, k)
  for i = 1, #words do
    r = self:get_sim_words(words[i], k)

    if r ~= nil then
      print("-------"..words[i].."-------")

      for j = 1, k do
        print(string.format("%s, %.4f", r[j][1], r[j][2]))
      end
    end
  end
end

-- split on separator
function Word2Vec:split(input, sep)
  if sep == nil then
    sep = "%s"
  end
  local t = {}; local i = 1
  for str in string.gmatch(input, "([^"..sep.."]+)") do
    t[i] = str; i = i + 1
  end
  return t
end

-- train the model using config parameters
function Word2Vec:train_model(corpus)
  self:train_stream(corpus)
end
