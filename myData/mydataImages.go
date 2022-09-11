package myData

import (
	"fmt"
	"image"
	_ "image/jpeg"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"

	"gonum.org/v1/gonum/mat"
)

func ReadImage(parallelism int, rank int, trainfrac float32) {
	// foldernametraining := "/home/si/Downloads/archive/seg_train/"
	foldernametraining := "../datasets/archive/seg_train/"
	ReadImageTraining(parallelism, rank, foldernametraining, trainfrac)

	foldernametesting := "../datasets/archive/seg_test/"
	ReadImageTesting(parallelism, rank, foldernametesting)

	fmt.Println("Parallel at ", rank)
	fmt.Println("The number of Training data is ", len(TrainingData))
	fmt.Println("The number of Training label is", len(TrainingTarget))
	fmt.Println("The number of Validing data is ", len(ValidingData))
	fmt.Println("The number of Validing lable is", len(ValidingTarget))
	fmt.Println("The number of Testing  data is ", len(TestingData))
	fmt.Println("The number of Testing  lable is", len(TestingTarget))
}

func ReadImageTraining(parallelism int, rank int, foldername string, trainfrac float32) {
	labels := []string{"buildings", "forest", "glacier", "mountain", "sea", "street"}
	datamaplabel := ReadImageMap(parallelism, rank, foldername)
	lendatamaplable := len(datamaplabel)
	lendatamaplabletrain := int(float32(lendatamaplable) * trainfrac)
	lendatamaplablevalid := int(float32(lendatamaplable) * (1 - trainfrac))

	for index := 0; index < lendatamaplabletrain; index++ {
		for k, v := range datamaplabel[index] {
			onehotitem := make([]float64, 6)
			for i := 0; i < 6; i++ {
				if k == labels[i] {
					onehotitem[i] = 1
				}
			}
			onehotitemDense := mat.NewDense(1, 6, onehotitem)
			TrainingTarget = append(TrainingTarget, *onehotitemDense)

			datatmp := LoadImageFile(v)
			datatmpDense := mat.NewDense(1, 150*150*3, datatmp)
			TrainingData = append(TrainingData, *datatmpDense)
		}
	}

	for index := 0; index < lendatamaplablevalid; index++ {
		for k, v := range datamaplabel[index+lendatamaplabletrain] {
			onehotitem := make([]float64, 6)
			for i := 0; i < 6; i++ {
				if k == labels[i] {
					onehotitem[i] = 1
				}
			}
			onehotitemDense := mat.NewDense(1, 6, onehotitem)
			ValidingTarget = append(ValidingTarget, *onehotitemDense)

			datatmp := LoadImageFile(v)
			datatmpDense := mat.NewDense(1, 150*150*3, datatmp)
			ValidingData = append(ValidingData, *datatmpDense)
		}
	}
}

func ReadImageTesting(parallelism int, rank int, foldername string) {
	labels := []string{"buildings", "forest", "glacier", "mountain", "sea", "street"}
	datamaplabel := ReadImageMap(parallelism, rank, foldername)
	for index := range datamaplabel {
		for k, v := range datamaplabel[index] {
			onehotitem := make([]float64, 6)
			for i := 0; i < 6; i++ {
				if k == labels[i] {
					onehotitem[i] = 1
				}
			}
			onehotitemDense := mat.NewDense(1, 6, onehotitem)
			TestingTarget = append(TestingTarget, *onehotitemDense)
			datatmp := LoadImageFile(v)
			datatmpDense := mat.NewDense(1, 150*150*3, datatmp)
			TestingData = append(TestingData, *datatmpDense)
		}
	}
}

func LoadImageFile(filename string) []float64 {
	ImageData := make([]float64, 150*150*3)
	reader, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer reader.Close()
	m, _, err := image.Decode(reader)
	if err != nil {
		log.Fatal(err)
	}

	maxr, _, _, _ := m.At(0, 0).RGBA()
	minr, _, _, _ := m.At(0, 0).RGBA()
	_, maxg, _, _ := m.At(0, 0).RGBA()
	_, ming, _, _ := m.At(0, 0).RGBA()
	_, _, maxb, _ := m.At(0, 0).RGBA()
	_, _, minb, _ := m.At(0, 0).RGBA()

	for x := 0; x < 150; x++ {
		for y := 0; y < 150; y++ {
			r, g, b, _ := m.At(x, y).RGBA()
			if r > maxr {
				maxr = r
			}
			if r < minr {
				minr = r
			}
			if g > maxg {
				maxg = g
			}
			if g < ming {
				ming = g
			}
			if b > maxb {
				maxb = b
			}
			if b < minb {
				minb = b
			}
		}
	}

	for x := 0; x < 150; x++ {
		for y := 0; y < 150; y++ {
			r, g, b, _ := m.At(x, y).RGBA()
			ImageData[x*150+y] = float64(r-minr) / float64(maxr-minr)
			ImageData[150*150+x*150+y] = float64(g-ming) / float64(maxg-ming)
			ImageData[150*150*2+x*150+y] = float64(b-minb) / float64(maxb-minb)
		}
	}
	return ImageData
}

func ReadImageMap(parallelism int, rank int, foldername string) []map[string]string {
	var filenames []string
	count := 0
	var label []string
	var datamaplabel []map[string]string
	err := filepath.Walk(foldername, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			fmt.Println(err)
			return err
		}
		if count%parallelism == rank {
			if strings.HasSuffix(path, "jpg") {
				filenames = append(filenames, path)
			}
		}
		count++
		return nil
	})
	if err != nil {
		fmt.Println(err)
	}

	for index := range filenames {
		pathform := strings.Split(filenames[index], "/")
		label = append(label, pathform[len(pathform)-2])

	}

	for index := range filenames {
		datamaplabletmp := map[string]string{label[index]: filenames[index]}
		datamaplabel = append(datamaplabel, datamaplabletmp)
	}

	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(datamaplabel), func(i, j int) { datamaplabel[i], datamaplabel[j] = datamaplabel[j], datamaplabel[i] })

	return datamaplabel
}
