library(hazer)
library(jpeg)
library(data.table)


pointreyes_dir <- "/Data_Img"

pointreyes_dirs <- dir(pointreyes_dir, ignore.case = TRUE, full.names = TRUE)

n_dirs <- length(pointreyes_dirs)


for(i in 1:n_dirs) {

	pointreyes_images <- dir(path = pointreyes_dirs[i],
                         ignore.case = TRUE, 
                         full.names = TRUE)

	n <- length(pointreyes_images)
	for(j in 1:n){
		image_path <- pointreyes_images[j]

		img <- jpeg::readJPEG(image_path)
		haze <- getHazeFactor(img)

		result = paste(image_path, "haze: ", haze[1], "A0: ", haze[2], "\n", sep=" ")
		write(result[1], file = "haze.txt", append = TRUE, , sep = "\n")
	}

}