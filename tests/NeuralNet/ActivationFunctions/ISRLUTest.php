<?php

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\ISRLU;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group ActivationFunctions
 * @covers \Rubix\ML\NeuralNet\ActivationFunctions\ISRLU
 */
class ISRLUTest extends TestCase
{
    /**
     * @var \Rubix\ML\NeuralNet\ActivationFunctions\ISRLU
     */
    protected $activationFn;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->activationFn = new ISRLU(1.0);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(ISRLU::class, $this->activationFn);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFn);
    }

    /**
     * @test
     * @dataProvider computeProvider
     *
     * @param \Tensor\Matrix $input
     * @param array[] $expected
     */
    public function compute(Matrix $input, array $expected) : void
    {
        $activations = $this->activationFn->compute($input)->asArray();

        $this->assertEquals($expected, $activations);
    }

    /**
     * @return \Generator<array>
     */
    public function computeProvider() : Generator
    {
        yield [
            Matrix::quick([
                [1.0, -0.5, 0.0, 20.0, -10.0],
            ]),
            [
                [1.0, -0.4472135954999579, 0.0, 20.0, -0.9950371902099892],
            ],
        ];

        yield [
            Matrix::quick([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [-0.11914522061843064, 0.31, -0.44001524777298334],
                [0.99, 0.08, -0.029986509105671005],
                [0.05, -0.46135273664198945, 0.54],
            ],
        ];
    }

    /**
     * @test
     * @dataProvider differentiateProvider
     *
     * @param \Tensor\Matrix $input
     * @param \Tensor\Matrix $activations
     * @param array[] $expected
     */
    public function differentiate(Matrix $input, Matrix $activations, array $expected) : void
    {
        $derivatives = $this->activationFn->differentiate($input, $activations)->asArray();

        $this->assertEquals($expected, $derivatives);
    }

    /**
     * @return \Generator<array>
     */
    public function differentiateProvider() : Generator
    {
        yield [
            Matrix::quick([
                [1.0, -0.5, 0.0, 20.0, -10.0],
            ]),
            Matrix::quick([
                [1.0, -0.4472135954999579, 0.0, 20.0, -0.9950371902099892],
            ]),
            [
                [1.0, 0.7155417527999326, 1.0, 1.0, 0.0009851853368415735],
            ],
        ];

        yield [
            Matrix::quick([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            Matrix::quick([
                [-0.11914522061843064, 0.31, -0.44001524777298334],
                [0.99, 0.08, -0.029986509105671005],
                [0.05, -0.46135273664198945, 0.54],
            ]),
            [
                [0.9787823723254356, 1.0, 0.7241273297133433],
                [1.0, 1.0, 0.9986515171569258],
                [1.0, 0.6983759455561986, 1.0],
            ],
        ];
    }
}
