
This module will take a JSON input string from stdin and produce the ouput as a JSON string to stdout. 

Please checkout sapphire/nlg/dstc2 and copy over the following directory (in diae2) to the same level as dstc2 (Verify the directory structure in diae2:/home/llajan/nlg/dstc2 in case of doubts): 

```
/home/llajan/nlg/data
```

To verify functionality do the following
```
cd dstc2/src/
th test.lua < json_input
```

During production time do the following
```
require 'interact'
NLG = NLG()
------------------
NLG:serve() # Calls service handler
```
