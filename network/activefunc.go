package network

/////////////////////Sigmod
func (nn *NeuralNetFrame) SigmoidActiveFuncInputLayer() {
	nn.dInputLayer.Apply(nn.Config.applySigmode, nn.dInputLayerTmp)
}

func (nn *NeuralNetFrame) SigmoidActiveFuncHiddenLayer() {
	nn.dHiddenLayer.Apply(nn.Config.applySigmode, nn.dHiddenLayerTmp)
}

func (nn *NeuralNetFrame) SigmoidActiveFuncOutputLayer() {
	nn.dOutputLayer.Apply(nn.Config.applySigmode, nn.dOutputLayerTmp)
}

func (nn *NeuralNetFrame) PrimeSigmoidActiveFuncInputLayer() {
	nn.dInputLayerSlope.Apply(nn.Config.applySigmodePrime, nn.dInputLayer)
	nn.dInputLayerSlope.MulElem(nn.dInputLayerSlope, nn.dInputLayer)
}

func (nn *NeuralNetFrame) PrimeSigmoidActiveFuncHiddenLayer() {
	nn.dHiddenLayerSlope.Apply(nn.Config.applySigmodePrime, nn.dHiddenLayer)
	nn.dHiddenLayerSlope.MulElem(nn.dHiddenLayerSlope, nn.dHiddenLayer)
}

func (nn *NeuralNetFrame) PrimeSigmoidActiveFuncOutputLayer() {
	nn.dOutputLayerSlope.Apply(nn.Config.applySigmodePrime, nn.dOutputLayer)
	nn.dOutputLayerSlope.MulElem(nn.dOutputLayerSlope, nn.dOutputLayer)
}

/////////////////////////Tanh
func (nn *NeuralNetFrame) TanhActiveFuncInputLayer() {
	nn.dInputLayer.Apply(nn.Config.applyTanh, nn.dInputLayerTmp)
}

func (nn *NeuralNetFrame) TanhActiveFuncHiddenLayer() {
	nn.dHiddenLayer.Apply(nn.Config.applyTanh, nn.dHiddenLayerTmp)
}

func (nn *NeuralNetFrame) TanhActiveFuncOutputLayer() {
	nn.dOutputLayer.Apply(nn.Config.applyTanh, nn.dOutputLayerTmp)
}

func (nn *NeuralNetFrame) PrimeTanhActiveFuncInputLayer() {
	nn.dInputLayerSlope.Apply(nn.Config.applyTanhPrime, nn.dInputLayerTmp)
}

func (nn *NeuralNetFrame) PrimeTanhActiveFuncHiddenLayer() {
	nn.dHiddenLayerSlope.Apply(nn.Config.applyTanhPrime, nn.dHiddenLayerTmp)
}

func (nn *NeuralNetFrame) PrimeTanhActiveFuncOutputLayer() {
	nn.dOutputLayerSlope.Apply(nn.Config.applyTanhPrime, nn.dOutputLayerTmp)
}

///////////////////Rule
func (nn *NeuralNetFrame) RuleActiveFuncInputLayer() {
	nn.dInputLayer.Apply(nn.Config.applyRule, nn.dInputLayerTmp)
}

func (nn *NeuralNetFrame) RuleActiveFuncHiddenLayer() {
	nn.dHiddenLayer.Apply(nn.Config.applyRule, nn.dHiddenLayerTmp)
}

func (nn *NeuralNetFrame) RuleActiveFuncOutputLayer() {
	nn.dOutputLayer.Apply(nn.Config.applyRule, nn.dOutputLayerTmp)
}

func (nn *NeuralNetFrame) PrimeRuleActiveFuncInputLayer() {
	nn.dInputLayerSlope.Apply(nn.Config.applyRulePrime, nn.dInputLayerTmp)
}

func (nn *NeuralNetFrame) PrimeRuleActiveFuncHiddenLayer() {
	nn.dHiddenLayerSlope.Apply(nn.Config.applyRulePrime, nn.dHiddenLayerTmp)
}

func (nn *NeuralNetFrame) PrimeRuleActiveFuncOutputLayer() {
	nn.dOutputLayerSlope.Apply(nn.Config.applyRulePrime, nn.dOutputLayerTmp)
}

func (nn *NeuralNetFrame) ApplySoftmaxAtEnd() {
	r, c := nn.dOutputLayer.Dims()
	for i := 0; i < r; i++ {
		tmpsum := 0.0
		for j := 0; j < c; j++ {
			tmpsum += nn.dOutputLayer.At(i, j)
		}
		for j := 0; j < c; j++ {
			nn.dOutputLayer.Set(i, j, nn.dOutputLayer.At(i, j)/tmpsum)
		}
	}
}
