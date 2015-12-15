require('nn')

-- Row-normalize a matrix
local function normalize(m)
  m_norm = torch.zeros(m:size())
  for i = 1, m:size(1) do
    m_norm[i] = m[i] / torch.norm(m[i])
  end
  return m_norm
end

local Word2VecSampler = torch.class("Word2VecSampler")

function Word2VecSampler:__init(model)
  self.word_vecs = model.word_vecs
  self.word2index = model.word2index
  self.index2word = model.index2word
  self.word_vecs_norm = normalize(self.word_vecs.weight:double())
end

function Word2VecSampler:save(path)
  local snap = {}

  snap.word_vecs =  self.word_vecs
  snap.word2index = self.word2index
  snap.index2word = self.index2word

  torch.save(path, snap)
end

function Word2VecSampler.load(path)
  return torch.load(path)
end

-- Return the k-nearest words to a word or a vector based on cosine similarity
-- w can be a string such as "king" or a vector for ("king" - "queen" + "man")
function Word2VecSampler:similar(w, k)
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

return Word2VecSampler
