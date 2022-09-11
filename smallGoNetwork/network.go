package main

import (
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidPrime(x float64) float64 {
	return x * (1.0 - x)
}

func softmax(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

type neuralNet struct {
	config  neuralNetConfig
	wHidden *mat.Dense
	bHidden *mat.Dense
	wOut    *mat.Dense
	bOut    *mat.Dense
}

type neuralNetConfig struct {
	inputNeurons  int
	outputNeurons int
	hiddenNeurons int
	numEpochs     int
	learningRate  float64
}

func main() {
	input := mat.NewDense(3, 4, []float64{
		1.0, 0.0, 1.0, 0.0,
		1.0, 0.0, 1.0, 1.0,
		0.0, 1.0, 0.0, 1.0,
	})
	lables := mat.NewDense(3, 1, []float64{1.0, 1.0, 0.0})
	config := neuralNetConfig{
		inputNeurons:  4,
		outputNeurons: 1,
		hiddenNeurons: 3,
		numEpochs:     5000,
		learningRate:  0.2,
	}

	network := newNetwork(config)
	if err := network.train(input, lables); err != nil {
		log.Fatal(err)
	}
	f := mat.Formatted(network.wHidden, mat.Prefix("        "))
	fmt.Printf("\nwHidden = %v\n\n", f)
	f = mat.Formatted(network.bHidden, mat.Prefix("        "))
	fmt.Printf("\nbHidden = %v\n\n", f)
	f = mat.Formatted(network.wOut, mat.Prefix("        "))
	fmt.Printf("\nwOut = %v\n\n", f)
	f = mat.Formatted(network.bOut, mat.Prefix("        "))
	fmt.Printf("\nbOut = %v\n\n", f)
}

func newNetwork(config neuralNetConfig) *neuralNet {
	return &neuralNet{config: config}
}

func (nn *neuralNet) train(x, y *mat.Dense) error {
	randSource := rand.NewSource(time.Now().Unix())
	randGen := rand.New(randSource)
	wHiddenRaw := make([]float64, nn.config.inputNeurons*nn.config.hiddenNeurons)
	bHiddenRaw := make([]float64, nn.config.hiddenNeurons)
	wOutRaw := make([]float64, nn.config.hiddenNeurons*nn.config.outputNeurons)
	bOutRaw := make([]float64, nn.config.outputNeurons)

	for _, param := range [][]float64{wHiddenRaw, bHiddenRaw, wOutRaw, bOutRaw} {
		for i := range param {
			param[i] = randGen.Float64()
		}
	}
	wHidden := mat.NewDense(nn.config.inputNeurons, nn.config.hiddenNeurons, wHiddenRaw)
	bHidden := mat.NewDense(1, nn.config.hiddenNeurons, bHiddenRaw)
	wOut := mat.NewDense(nn.config.hiddenNeurons, nn.config.outputNeurons, wOutRaw)
	bOut := mat.NewDense(1, nn.config.outputNeurons, bOutRaw)

	output := &mat.Dense{}
	for i := 0; i < nn.config.numEpochs; i++ {
		hiddenlayerInput := &mat.Dense{}
		hiddenlayerInput.Mul(x, wHidden)
		addBHidden := func(_, col int, v float64) float64 { return v + bHidden.At(0, col) }
		hiddenlayerInput.Apply(addBHidden, hiddenlayerInput)

		hiddenLayerActivations := &mat.Dense{}
		applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
		hiddenLayerActivations.Apply(applySigmoid, hiddenlayerInput)

		outputLayerInput := &mat.Dense{}
		outputLayerInput.Mul(hiddenLayerActivations, wOut)
		addBout := func(_, col int, v float64) float64 { return v + bOut.At(0, col) }
		outputLayerInput.Apply(addBout, outputLayerInput)

		output.Apply(applySigmoid, outputLayerInput)

		networkError := &mat.Dense{}
		networkError.Sub(y, output)

		slopeOutputLayer := &mat.Dense{}
		applySigmoidPrime := func(_, _ int, v float64) float64 { return sigmoidPrime(v) }
		slopeOutputLayer.Apply(applySigmoidPrime, output)

		slopeHiddenLayer := &mat.Dense{}
		slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)

		dOutput := &mat.Dense{}
		dOutput.MulElem(networkError, slopeOutputLayer)
		errorAtHiddenLayer := &mat.Dense{}
		errorAtHiddenLayer.Mul(dOutput, wOut.T())

		dHiddenLayer := &mat.Dense{}
		dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)

		wOutAdj := &mat.Dense{}
		wOutAdj.Mul(hiddenLayerActivations.T(), dOutput)
		wOutAdj.Scale(nn.config.learningRate, wOutAdj)
		wOut.Add(wOut, wOutAdj)

		bOutAdj, err := sumAlongAxis(0, dOutput)
		if err != nil {
			return err
		}
		bOutAdj.Scale(nn.config.learningRate, bOutAdj)
		bOut.Add(bOut, bOutAdj)

		wHiddenAdj := &mat.Dense{}
		wHiddenAdj.Mul(x.T(), dHiddenLayer)
		wHiddenAdj.Scale(nn.config.learningRate, wHiddenAdj)
		wHiddenAdj.Add(wHidden, wHiddenAdj)

		bHiddenAdj, err := sumAlongAxis(0, dHiddenLayer)
		if err != nil {
			return err
		}
		bHiddenAdj.Scale(nn.config.learningRate, bHiddenAdj)
		bHidden.Add(bHidden, bHiddenAdj)
	}
	nn.wHidden = wHidden
	nn.bHidden = bHidden
	nn.wOut = wOut
	nn.bOut = bOut
	return nil
}

func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {
	numRows, numCols := m.Dims()
	var output *mat.Dense
	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)

	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(1, numRows, data)
	default:
		return nil, errors.New("invalid axis, must be 0 or 1")
	}
	return output, nil
}
