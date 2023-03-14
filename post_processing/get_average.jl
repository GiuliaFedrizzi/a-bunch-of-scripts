"""
Reads the first file (my_experiment00000.csv)
takes the average od all porosity values
builds a dataframe
prints values
plots them (or saves them)
"""

using CSV
using DataFrames
using Statistics
using Glob
using Plots
import RecipesBase: plot

println("starting...")
rad_frac_array = []
avPhi_array = []
# list all directories that start with lp
dirList = filter(isdir,readdir(glob"por*"))  # name of directories
for d in dirList
	myExp = CSV.File(d*"/my_experiment00000.csv")   # take the file with that name in every directory
	averagePhi=mean(skipmissing(myExp.Porosity))    # calculate mean porosity
	println(d," ",averagePhi)  
	rad_frac=chop(d,head = 3,tail = 0)                # get rad_frac from the directory name
	rad_frac=parse(Float64,rad_frac)/100.0            # rad_frac is string, parse to float and divide by 100 to get original number
	append!(rad_frac_array,rad_frac)                # append the new value to a big array with all the rad_frac values
	append!(avPhi_array,averagePhi)                # append the new value to a big array with all the phi values
end

rad_phi=DataFrame("radFrac"=>rad_frac_array,"avPhi"=>avPhi_array)
sort!(rad_phi)
# println(rad_frac_array)
# println(avPhi_array)

println(rad_phi)

p=plot(rad_phi[!,:radFrac],rad_phi[!,:avPhi],label = "Average Porosity",seriestype = :scatter,dpi=300)
xlabel!("rad_frac")
ylabel!("average phi")
gui(p)
readline()   # wait for input so the plot stays open
# png("phiPlot")