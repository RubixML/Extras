<?php

namespace Rubix\ML\Tests\NeuralNet\ActivationFunctions;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\ISRU;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group ActivationFunctions
 * @covers \Rubix\ML\NeuralNet\ActivationFunctions\ISRU
 */
class ISRUTest extends TestCase
{
    /**
     * @var \Rubix\ML\NeuralNet\ActivationFunctions\ISRU
     */
    protected $activationFn;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->activationFn = new ISRU(1.0);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(ISRU::class, $this->activationFn);
        $this->assertInstanceOf(ActivationFunction::class, $this->activationFn);
    }

    /**
     * @test
     * @dataProvider computeProvider
     *
     * @param \Tensor\Matrix $input
     * @param array<array<float>> $expected
     */
    public function compute(Matrix $input, array $expected) : void
    {
        $activations = $this->activationFn->activate($input)->asArray();

        $this->assertEquals($expected, $activations);
    }

    /**
     * @return \Generator<array<mixed>>
     */
    public function computeProvider() : Generator
    {
        yield [
            Matrix::quick([
                [1.0, -0.5, 0.0, 20.0, -10.0],
            ]),
            [
                [0.7071067811865475, -0.4472135954999579, 0.0, 0.9987523388778445, -0.9950371902099892],
            ],
        ];

        yield [
            Matrix::quick([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            [
                [-0.11914522061843064, 0.29609877111408056, -0.44001524777298334],
                [0.7035445979201053, 0.07974522228289, -0.029986509105671005],
                [0.04993761694389223, -0.46135273664198945, 0.47514891473488396],
            ],
        ];
    }

    /**
     * @test
     * @dataProvider differentiateProvider
     *
     * @param \Tensor\Matrix $input
     * @param \Tensor\Matrix $activations
     * @param array<array<mixed>> $expected
     */
    public function differentiate(Matrix $input, Matrix $activations, array $expected) : void
    {
        $derivatives = $this->activationFn->differentiate($input, $activations)->asArray();

        $this->assertEquals($expected, $derivatives);
    }

    /**
     * @return \Generator<array<mixed>>
     */
    public function differentiateProvider() : Generator
    {
        yield [
            Matrix::quick([
                [1.0, -0.5, 0.0, 20.0, -10.0],
            ]),
            Matrix::quick([
                [0.7071067811865475, -0.44721359549995794, 0.0, 0.9987523388778445, -0.9950371902099892],
            ]),
            [
                [0.3535533905932737, 0.7155417527999326, 1.0, 0.0001245327105832724, 0.0009851853368415735],
            ],
        ];

        yield [
            Matrix::quick([
                [-0.12, 0.31, -0.49],
                [0.99, 0.08, -0.03],
                [0.05, -0.52, 0.54],
            ]),
            Matrix::quick([
                [-0.11914522061843064, 0.29609877111408056, -0.44001524777298334],
                [0.7035445979201053, 0.07974522228289, -0.029986509105671005],
                [0.04993761694389223, -0.46135273664198945, 0.47514891473488396],
            ]),
            [
                [0.978782372325436, 0.8714144021297815, 0.7241273297133433],
                [0.3588965754306386, 0.9904762306599015, 0.9986515171569258],
                [0.9962616846661791, 0.6983759455561989, 0.6812522434632957],
            ],
        ];
    }
}
