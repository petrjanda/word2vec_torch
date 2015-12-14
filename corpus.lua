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
end


