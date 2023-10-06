
function key_name=get_keyname(image_name,prefix)
        key_name=image_name;
        key_name=strrep(key_name,'-','_');
        key_name=erase(key_name,'.tif');
        key_name=strcat(prefix,key_name);
end