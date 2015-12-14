-- Split the text by the given separator
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

-- Corpus class for loading text files as the NLP tasks
local Corpus = torch.class("Corpus")

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

-- Stream corpus sentenses (lines split by space)
function Corpus:streamSentenses(path, fn)
  local f = io.open(path, "r")

  for line in f:lines() do
    sentence = split(line)

    fn(sentence)
  end

  f:close()
end

-- Filter words which occur with some minimal frequency
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

  local t = torch.IntTensor(tableSize)
  local word_index = 1

  -- probability of the word in the corpus
  local word_prob = self.vocab[self.index2word[word_index]]^alpha / total_count_pow

  for idx = 1, tableSize do
    t[idx] = word_index

    if idx / tableSize > word_prob then
      word_index = word_index + 1

      if(word_index <= vocab_size) then
        word_prob = word_prob + self.vocab[self.index2word[word_index]] ^ alpha / total_count_pow
      end
    end

    if word_index > vocab_size then
      word_index = word_index - 1
    end
  end

  return t
end
