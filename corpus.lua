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
