"""
This script builds an HDF5 file out of the training data for
State Farm Distracted Driver Detection
"""

using HDF5
using Colors
using Images
using ImageMagick

searchdir(path,key) = filter(x->contains(x,key), readdir(path))

#Get all the file names.
#Folder denotes the image class
myFiles=[]
labels=[]

for i=0:9
    files=searchdir("train/c$i/","jpg")
    for ff in files
        push!(myFiles,abspath(ff))
        push!(labels,i)
    end
end


#fix constats
n_data=size(myFiles)[1]
w=
h=

#open hdf5 data objs
h5open("train.hdf5", "w") do h5
    dset_data = d_create(h5, "data", datatype(Float32), dataspace(w, h, 1, n_data))
    dset_label = d_create(h5, "label", datatype(Int64), dataspace(1, n_data))


# for each image set load and store it in the hdf5 data object
for i in 1:size(myFiles)[1]
    
    img=load(imgFiles[i]);
    img=convert(Array{Gray},img);
    
    #may need to convert
    dset_data[:,:,1,i] = img
    dset_label[1,i] = labels[i]
end

#write out files
close(label_f)
close(data_f)