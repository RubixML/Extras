<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Transformers\Elastic;
use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\DeltaTfIdfTransformer;
use PHPUnit\Framework\TestCase;
use RuntimeException;

class DeltaTfIdfTransformerTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Labeled
     */
    protected $dataset;

    /**
     * @var \Rubix\ML\Transformers\DeltaTfIdfTransformer
     */
    protected $transformer;

    /**
     * @before
     */
    public function setUp() : void
    {
        $this->dataset = new Labeled([
            [1, 2, 0, 0, 1, 0, 0, 0, 1, 2, 0, 2],
            [1, 1, 0, 0, 2, 0, 0, 1, 4, 2, 0, 0],
            [1, 0, 0, 0, 0, 3, 1, 1, 1, 0, 1, 0],
            [0, 1, 1, 0, 0, 2, 2, 1, 0, 0, 0, 3],
            [0, 0, 2, 1, 0, 1, 1, 1, 0, 0, 0, 1],
        ], ['pos', 'pos', 'pos', 'neg', 'neg']);

        $this->transformer = new DeltaTfIdfTransformer();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(DeltaTfIdfTransformer::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
        $this->assertInstanceOf(Stateful::class, $this->transformer);
        $this->assertInstanceOf(Elastic::class, $this->transformer);
    }

    /**
     * @test
     */
    public function fitTransform() : void
    {
        $this->transformer->fit($this->dataset);

        $this->dataset->apply($this->transformer);

        $outcome = [
            [1.752038698388137, 3.355961350017359, 0, 0, 2.039720770839918, 0, 0, 0, 1.6834522722589234, 3.7081819436067107, 0, 3.4143951825402823],
            [1.752038698388137, 1.9504962419091942, 0, 0, 3.7328679513998635, 0, 0, 1.7953123053131437, 5.899847596583417, 3.7081819436067107, 0, 0],
            [1.752038698388137, 0, 0, 0, 0, 4.855956224822119, 1.9504962419091942, 1.7953123053131437, 1.6834522722589234, 0, 2.4451858789480827, 0],
            [0, 1.9504962419091942, 2.039720770839918, 0, 0, 3.450491116713955, 3.355961350017359, 1.7953123053131437, 0, 0, 0, 4.819860290648447],
            [0, 0, 3.7328679513998635, 2.4451858789480827, 0, 2.0450260086057903, 1.9504962419091942, 1.7953123053131437, 0, 0, 0, 2.008930074432118],
        ];

        $this->assertEquals($outcome, $this->dataset->samples());
    }

    /**
     * @test
     */
    public function transformUnfitted() : void
    {
        $this->expectException(RuntimeException::class);

        $samples = $this->dataset->samples();

        $this->transformer->transform($samples);
    }
}
