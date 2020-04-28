function test(("x"))
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
   flabel = Meta.parse("f$v")
   val = Meta.parse("$v")
   fn = fieldnames(typeof(gmres))
   fn = Meta.parse("$fn[1]")
   println(val)
   println(typeof(flabel))
   @eval $flabel($fn) = $fn + $val
end
