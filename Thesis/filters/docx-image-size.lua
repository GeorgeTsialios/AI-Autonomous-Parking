-- TODO

-- function Image(img)
--     -- get dimensions
--     img.attributes.width = size_sep[1]
--     img.attributes.height = size_sep[2]
--     local width, height = nil, nil -- implement
--     img.attributes.width = width
--     img.attributes.height = height
--     return img
-- end


funFigure = function(element)
    local attrfigure = pandoc.Attr( "", { }, {{"custom-style", "Picture" }} )
    local attrfiguretext = pandoc.Attr( "", { }, {{"custom-style", "Picture Caption" }} )
    local figuretext = pandoc.Div(pandoc.Para(element.content[1].caption), attrfiguretext)
    element.content[1].caption = pandoc.Null
    local figure = pandoc.Div(element, attrfigure)
    return { figure, figuretext }
end