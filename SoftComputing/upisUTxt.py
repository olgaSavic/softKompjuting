import sys
import program

file = open('out.txt','w')
file.write("RA 144/2015 Olga Savic")
file.write('\nfile  sum')

r0 = program.main("video-0.avi") # main vraca sumu
file.write("\nvideo-0.avi\t" + str(r0))

r1 = program.main("video-1.avi") # main vraca sumu
file.write("\nvideo-1.avi\t" + str(r1))

r2 = program.main("video-2.avi") # main vraca sumu
file.write("\nvideo-2.avi\t" + str(r2))

r3 = program.main("video-3.avi") # main vraca sumu
file.write("\nvideo-3.avi\t" + str(r3))

r4 = program.main("video-4.avi") # main vraca sumu
file.write("\nvideo-4.avi\t" + str(r4))

r5 = program.main("video-5.avi") # main vraca sumu
file.write("\nvideo-5.avi\t" + str(r5))

r6 = program.main("video-6.avi") # main vraca sumu
file.write("\nvideo-6.avi\t" + str(r6))

r7 = program.main("video-7.avi") # main vraca sumu
file.write("\nvideo-7.avi\t" + str(r7))

r8 = program.main("video-8.avi") # main vraca sumu
file.write("\nvideo-8.avi\t" + str(r8))

r9 = program.main("video-9.avi") # main vraca sumu
file.write("\nvideo-9.avi\t" + str(r9))

    

