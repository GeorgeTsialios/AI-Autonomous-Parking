function startswith(text, prefix)
  return text:find(prefix, 1, true) == 1
end

function fix_path (path)
  return 'images/' .. path

end

function Link (element)
  element.target = element.target
  return element
end

function Image (element)
  if startswith(element.src, "https://") then
    -- print(element.src)
    element.src = ' ' .. element.src
    -- print(element.src)
    -- print("------------------------")
  else
    element.src = fix_path(element.src)
  end
  return element
end
  