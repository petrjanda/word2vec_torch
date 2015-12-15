
function load(action)
  local f = io.open("corpus.txt", "r")
  local i = 0

  for line in f:lines() do
    action(line, i)

    i = i + 1
  end

  f:close()
end

function main()
  function process(line, num)
    print(num, line)
  end

  load(process)
end

main()
