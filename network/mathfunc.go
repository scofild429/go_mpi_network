package network

import (
	"errors"
	"math"

	"github.com/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

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

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidPrime(x float64) float64 {
	return (1.0 - x)

}

func rule(x float64) float64 {
	if x < 0 {
		return 0
	} else {
		return 0.5 * x
	}
}

func rulePrime(x float64) float64 {
	if x < 0 {
		return 1
	} else {
		return 0.5
	}
}

func tanh(x float64) float64 {
	return math.Tanh(x)
}

func tanhPrime(x float64) float64 {
	return 1 - math.Pow(2, math.Tanh(x))
}
