package network

import "math"

func (nn *NeuralNetFrame) NormalizationInputLayer() {
	r, c := nn.dInputLayerTmp.Dims()
	max := nn.dInputLayerTmp.At(0, 0)
	min := nn.dInputLayerTmp.At(0, 0)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if nn.dInputLayerTmp.At(i, j) > max {
				max = nn.dInputLayerTmp.At(i, j)
			}
			if nn.dInputLayerTmp.At(i, j) < min {
				min = nn.dInputLayerTmp.At(i, j)
			}
		}
	}
	normalization := func(_, _ int, v float64) float64 { return (v - min) / (max - min) }
	nn.dInputLayerTmp.Apply(normalization, nn.dInputLayerTmp)
}

func (nn *NeuralNetFrame) StandardizationInputLayer() {
	r, c := nn.dInputLayerTmp.Dims()
	tempmean := 0.0
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			tempmean += nn.dInputLayerTmp.At(i, j)
		}
	}
	mean := tempmean / float64(r*c)
	tempstd := 0.0
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			tempstd += (nn.dInputLayerTmp.At(i, j) - mean) * (nn.dInputLayerTmp.At(i, j) - mean)
		}
	}
	std := math.Sqrt(tempstd) / float64(r*c)
	standardization := func(_, _ int, v float64) float64 { return (v - mean) / std }
	nn.dInputLayerTmp.Apply(standardization, nn.dInputLayerTmp)

}

func (nn *NeuralNetFrame) NormalizationHiddenLayer() {
	r, c := nn.dHiddenLayerTmp.Dims()
	max := nn.dHiddenLayerTmp.At(0, 0)
	min := nn.dHiddenLayerTmp.At(0, 0)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if nn.dHiddenLayerTmp.At(i, j) > max {
				max = nn.dHiddenLayerTmp.At(i, j)
			}
			if nn.dHiddenLayerTmp.At(i, j) < min {
				min = nn.dHiddenLayerTmp.At(i, j)
			}
		}
	}
	normalization := func(_, _ int, v float64) float64 { return (v - min) / (max - min) }
	nn.dHiddenLayerTmp.Apply(normalization, nn.dHiddenLayerTmp)
}

func (nn *NeuralNetFrame) StandardizationHiddenLayer() {
	r, c := nn.dHiddenLayerTmp.Dims()
	tempmean := 0.0
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			tempmean += nn.dHiddenLayerTmp.At(i, j)
		}
	}
	mean := tempmean / float64(r*c)
	tempstd := 0.0
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			tempstd += math.Pow(2, (nn.dHiddenLayerTmp.At(i, j) - mean))
		}
	}
	std := math.Sqrt(tempstd) / float64(r*c)
	standardization := func(_, _ int, v float64) float64 { return (v - mean) / std }
	nn.dHiddenLayerTmp.Apply(standardization, nn.dHiddenLayerTmp)
}

func (nn *NeuralNetFrame) NormalizationOutputLayer() {
	r, c := nn.dOutputLayerTmp.Dims()
	max := nn.dOutputLayerTmp.At(0, 0)
	min := nn.dOutputLayerTmp.At(0, 0)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if nn.dOutputLayerTmp.At(i, j) > max {
				max = nn.dOutputLayerTmp.At(i, j)
			}
			if nn.dOutputLayerTmp.At(i, j) < min {
				min = nn.dOutputLayerTmp.At(i, j)
			}
		}
	}
	normalization := func(_, _ int, v float64) float64 { return (v - min) / (max - min) }
	nn.dOutputLayerTmp.Apply(normalization, nn.dOutputLayerTmp)
}

func (nn *NeuralNetFrame) StandardizationOutputLayer() {
	r, c := nn.dOutputLayerTmp.Dims()
	tempmean := 0.0
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			tempmean += nn.dOutputLayerTmp.At(i, j)
		}
	}
	mean := tempmean / float64(r*c)
	tempstd := 0.0
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			tempstd += math.Pow(2, (nn.dOutputLayerTmp.At(i, j) - mean))
		}
	}
	std := math.Sqrt(tempstd) / float64(r*c)
	standardization := func(_, _ int, v float64) float64 { return (v - mean) / std }
	nn.dOutputLayerTmp.Apply(standardization, nn.dOutputLayerTmp)
}
