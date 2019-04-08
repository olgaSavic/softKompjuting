import sys
import program

file = open('out0.txt','w')
file.write("RA 144/2015 Olga Savic")
file.write('\nfile  sum')

r0 = program.main("video-0.avi") # main vraca sumu
file.write("\nvideo-0.avi\t" + str(r0))

