<?php

namespace Rubix\ML\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\ActivationFunctions\SELU;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

/**
 * Alpha Dropout
 *
 * Alpha Dropout is a type of dropout layer that maintains the mean and variance
 * of the original inputs in order to ensure the self-normalizing property of
 * SELU networks with dropout. Alpha Dropout fits with SELU networks by randomly
 * setting activations to the negative saturation value of the activation function
 * at a given ratio each pass.
 *
 * References:
 * [1] G. Klambauer et al. (2017). Self-Normalizing Neural Networks.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class AlphaDropout implements Hidden
{
    /**
     * The negative saturation value of SELU.
     *
     * @var float
     */
    protected const ALPHA_P = -SELU::ALPHA * SELU::SCALE;

    /**
     * The ratio of neurons that are dropped during each training pass.
     *
     * @var float
     */
    protected $ratio;

    /**
     * The affine transformation scaling coefficient.
     *
     * @var float
     */
    protected $alpha;

    /**
     * The affine transformation centering coefficient.
     *
     * @var float
     */
    protected $beta;

    /**
     * The width of the layer.
     *
     * @var int|null
     */
    protected $width;

    /**
     * The memoized dropout mask.
     *
     * @var \Tensor\Matrix|null
     */
    protected $mask;

    /**
     * @param float $ratio
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(float $ratio = 0.1)
    {
        if ($ratio <= 0.0 or $ratio >= 1.0) {
            throw new InvalidArgumentException('Ratio must be strictly'
                . " between 0 and 1, $ratio given.");
        }

        $this->ratio = $ratio;
        $this->alpha = ((1.0 - $ratio) * (1.0 + $ratio * self::ALPHA_P ** 2)) ** -0.5;
        $this->beta = -$this->alpha * self::ALPHA_P * $ratio;
    }

    /**
     * Return the width of the layer.
     *
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return int
     */
    public function width() : int
    {
        if (!$this->width) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        return $this->width;
    }

    /**
     * Initialize the layer with the fan in from the previous layer and return
     * the fan out for this layer.
     *
     * @param int $fanIn
     * @return int
     */
    public function initialize(int $fanIn) : int
    {
        $fanOut = $fanIn;

        $this->width = $fanOut;

        return $fanOut;
    }

    /**
     * Compute a forward pass through the layer.
     *
     * @param \Tensor\Matrix $input
     * @return \Tensor\Matrix
     */
    public function forward(Matrix $input) : Matrix
    {
        $mask = Matrix::rand(...$input->shape())
            ->greater($this->ratio);

        $saturation = $mask->map([$this, 'saturate']);

        $this->mask = $mask;

        return $input->multiply($mask)
            ->add($saturation)
            ->multiply($this->alpha)
            ->add($this->beta);
    }

    /**
     * Compute an inferential pass through the layer.
     *
     * @param \Tensor\Matrix $input
     * @return \Tensor\Matrix
     */
    public function infer(Matrix $input) : Matrix
    {
        return $input;
    }

    /**
     * Calculate the gradients of the layer and update the parameters.
     *
     * @param \Rubix\ML\Deferred $prevGradient
     * @param \Rubix\ML\NeuralNet\Optimizers\Optimizer $optimizer
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return \Rubix\ML\Deferred
     */
    public function back(Deferred $prevGradient, Optimizer $optimizer) : Deferred
    {
        if (!$this->mask) {
            throw new RuntimeException('Must perform forward pass before'
                . ' backpropagating.');
        }

        $mask = $this->mask;

        unset($this->mask);

        return new Deferred([$this, 'gradient'], [$prevGradient, $mask]);
    }

    /**
     * Calculate the gradient for the previous layer.
     *
     * @param \Rubix\ML\Deferred $prevGradient
     * @param \Tensor\Matrix $mask
     * @return \Tensor\Matrix
     */
    public function gradient(Deferred $prevGradient, Matrix $mask)
    {
        return $prevGradient()->multiply($mask);
    }

    /**
     * Boost dropped neurons by a factor of alpha p.
     *
     * @param int $value
     * @return float
     */
    public function saturate(int $value) : float
    {
        return $value == 0 ? self::ALPHA_P : 0.0;
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Alpha Dropout (ratio: {$this->ratio})";
    }
}
