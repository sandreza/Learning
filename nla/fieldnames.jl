function test((x,y,z)...)
    println(y)
end

fn = fieldnames(typeof(gmres))


a = (x,y,z)
function test2(a...)
    println(y)
end


function test3(:($fn))
    println(atol)
end

for v in [:1, :2, :3, :4, :5]
   plabel = Meta.parse("p$v")
   val = Meta.parse("$v")
   @eval $plabel(x) = x + $val
end
