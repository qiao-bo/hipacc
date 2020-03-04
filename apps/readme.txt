Application folder for new samples

How to enable stream-based code generation for apps:
1 follow the standard getting started guide at (https://hipacc-lang.org/install.html)
2 the apps should be copied automatically to the same directory as samples  
3 enable the Hipacc flag `-use-stream n` in cuda.conf
4 `-use-stream 1` for single-stream and `-use-stream 2` for multi-stream
