package myData

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"time"

	mapset "github.com/deckarep/golang-set"
	"github.com/go-gota/gota/dataframe"
	"gonum.org/v1/gonum/mat"
)

var ALLDATA []allData

type allData struct {
	data         mat.Dense
	targetname   string
	target       int
	targetonehot mat.Dense
}

func Onehot(data dataframe.DataFrame) ([]int, [][]float64) {
	array := data.Records()
	dataset := mapset.NewSet()
	r, _ := data.Dims()
	for i := 1; i < r; i++ { // header is over jump from staring with 1
		dataset.Add(array[i][0])
	}
	dataslice := dataset.ToSlice()
	sort.Slice(dataslice, func(i, j int) bool {
		return fmt.Sprint(dataslice[i]) < fmt.Sprint(dataslice[j])
	})

	// fmt.Println(dataslice)
	var onehotarray [][]float64
	onehotnumber := []int{}
	len := dataset.Cardinality()
	for i := range array {
		onehotitem := make([]float64, len)
		// onehotitem := []int{0, 0, 0}
		//fmt.Println(onehotitem)
		for k := range dataslice {
			if dataslice[k] == array[i][0] {
				onehotnumber = append(onehotnumber, k)
				onehotitem[k] = 1
				onehotarray = append(onehotarray, onehotitem)
			}
		}
	}
	return onehotnumber, onehotarray
}

type MyMatrix struct {
	dataframe.DataFrame
}

func (m MyMatrix) Dims() (r, c int) {
	r, c = m.DataFrame.Dims()
	return r, c
}

func (m MyMatrix) At(i, j int) float64 {
	ele := m.DataFrame.Elem(i, j).Float()
	return ele
}

func (m MyMatrix) T() mat.Matrix {
	return mat.Transpose{Matrix: m}
}

func LoadDataIris(trainratio float64, validratio float64, testratio float64, parallelism int, rank int) {
	filename := "../datasets/iris.csv"
	file, err := os.Open(filename)
	if err != nil {
		fmt.Println(err)
	}
	defer file.Close()
	dff := dataframe.ReadCSV(file)
	// fmt.Println("orignal", df)
	datalen := dff.Nrow()
	var subindex = []int{}
	for i := 0; i < datalen-parallelism; i += parallelism {
		subindex = append(subindex, int(i+rank))
	}

	// fmt.Println(subindex)
	df := dff.Subset(subindex)
	//	fmt.Println("spileted", df)
	target := df.Select("variety")
	onehotnumber, onehotarray := Onehot(target)
	targetnames := target.Records()
	// fmt.Println(onehotnumber)

	var matrix MyMatrix
	df = df.Drop("variety")
	matrix.DataFrame = df

	r, c := matrix.Dims()
	dataarrays := []mat.Dense{}
	targetarrays := []mat.Dense{}
	for i := 0; i < r; i++ {
		dataarray := mat.Row(nil, i, matrix)
		for index, item := range dataarray {
			dataarray[index] = item
		}
		//	fmt.Println(dataarray)
		rowdatadense := mat.NewDense(1, c, dataarray)
		dataarrays = append(dataarrays, *rowdatadense)

		rowtargetarray := mat.NewDense(1, 3, onehotarray[i])
		targetarrays = append(targetarrays, *rowtargetarray)
	}
	for i := 0; i < r; i++ {
		var dataitem allData
		dataitem.data = dataarrays[i]
		dataitem.targetname = targetnames[i][0]
		dataitem.target = onehotnumber[i]
		dataitem.targetonehot = targetarrays[i]
		ALLDATA = append(ALLDATA, dataitem)
	}

	dataLen := len(ALLDATA)
	//	fmt.Println("hi", dataLen)
	rand.Seed(time.Now().Unix())
	rand.Shuffle(dataLen, func(i, j int) { ALLDATA[i], ALLDATA[j] = ALLDATA[j], ALLDATA[i] })

	trainingNum := int(math.Floor(float64(dataLen) * trainratio))
	validNum := int(math.Floor(float64(dataLen) * validratio))
	testNum := int(math.Floor(float64(dataLen) * testratio))

	for i := 0; i < trainingNum; i++ {
		TrainingData = append(TrainingData, ALLDATA[i].data)
		TrainingTarget = append(TrainingTarget, ALLDATA[i].targetonehot)
	}

	for i := trainingNum; i < trainingNum+testNum; i++ {
		TestingData = append(TestingData, ALLDATA[i].data)
		TestingTarget = append(TestingTarget, ALLDATA[i].targetonehot)
	}

	for i := trainingNum + testNum; i < trainingNum+testNum+validNum; i++ {
		// fmt.Println(ALLDATA[i].data)
		ValidingData = append(ValidingData, ALLDATA[i].data)
		ValidingTarget = append(ValidingTarget, ALLDATA[i].targetonehot)
	}
	fmt.Println("In Parallelism", rank)
	fmt.Println("The number of Training data is ", len(TrainingData))
	fmt.Println("The number of Training label is", len(TrainingTarget))
	fmt.Println("The number of Validing data is ", len(ValidingData))
	fmt.Println("The number of Validing lable is", len(ValidingTarget))
	fmt.Println("The number of Testing  data is ", len(TestingData))
	fmt.Println("The number of Testing  lable is", len(TestingTarget))
}
