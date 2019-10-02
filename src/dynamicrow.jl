dynamicrow(dict) = DynamicObject(dict)

struct DynamicObject{T}
    dict::T
end

_dict(obj) = getfield(obj, :dict)

Base.propertynames(obj::DynamicObject) = sort!(collect(keys(_dict(obj))))
Base.getproperty(obj::DynamicObject, name::Symbol) =
    _dict(obj)[keytype(_dict(obj))(name)]
