local Corpus = torch.class("Corpus")

local function split(input, sep)
  if sep == nil then
    sep = "%s"
  end

  local t = {}
  local i = 1

  for str in string.gmatch(input, "([^"..sep.."]+)") do
    t[i] = str
    i = i + 1
  end

  return t
end


function Corpus:__init()
  self.total = 0
  self.lines = 0
  self.vocab = {}

  self.index2word = {}
  self.word2index = {}
end

function Corpus:read(path)
  local f = io.open(path, "r")

  for line in f:lines() do
    for _, word in ipairs(split(line)) do
      if self.vocab[word] == nil then
        self.vocab[word] = 1	 
      else
        self.vocab[word] = self.vocab[word] + 1
      end

      self.total = self.total + 1
    end

    self.lines = self.lines + 1
  end

  f:close()
end

function Corpus:filter(minfreq)
  for word, count in pairs(self.vocab) do
    if count < minfreq then
      self.vocab[word] = nil
    end
  end
end

function Corpus:buildIndices()
  for word, count in pairs(self.vocab) do
    self.index2word[#self.index2word + 1] = word
    self.word2index[word] = #self.index2word
  end
end

function Corpus:buildUnigramsTable(alpha, tableSize)
  local vocab_size = #self.index2word
  local total_count_pow = 0

  for _, count in pairs(self.vocab) do
    total_count_pow = total_count_pow + count^alpha
  end   

  local table = torch.IntTensor(tableSize)
  local word_index = 1
  local word_prob = self.vocab[self.index2word[word_index]]^alpha / total_count_pow

  for idx = 1, tableSize do
    table[idx] = word_index

    if idx / tableSize > word_prob then
      word_index = word_index + 1
      word_prob = word_prob + self.vocab[self.index2word[word_index]]^alpha / total_count_pow
    end

    if word_index > vocab_size then
      word_index = word_index - 1
    end
  end

  return table
end
