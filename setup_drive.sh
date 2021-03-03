diskutil umount /Volumes/Cuore 
sshfs -o reconnect -o volname=Cuore yocum@cuore-login.lngs.infn.it:/nfs/cuore1/scratch/yocum/ /Volumes/Cuore
