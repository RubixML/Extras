<?php

namespace Rubix\ML\Benchmarks\Transformers;

use Tensor\Matrix;
use Tensor\Vector;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Transformers\DeltaTfIdfTransformer;

/**
 * @Groups({"Transformers"})
 * @BeforeMethods({"setUp"})
 */
class DeltaTfIdfTransformerBench
{
    protected const DATASET_SIZE = 10000;

    /**
     * @var \Rubix\ML\Datasets\Labeled
     */
    protected $dataset;

    /**
     * @var \Rubix\ML\Transformers\DeltaTfIdfTransformer
     */
    protected $transformer;

    public function setUp() : void
    {
        $mask = Matrix::rand(self::DATASET_SIZE, 4)
            ->greater(0.8);

        $samples = Matrix::gaussian(self::DATASET_SIZE, 4)
            ->multiply($mask)
            ->asArray();

        $labels = Vector::rand(self::DATASET_SIZE)
            ->greater(0.5)
            ->asArray();

        $this->dataset = Labeled::quick($samples, $labels);

        $this->transformer = new DeltaTfIdfTransformer(1.0);
    }

    /**
     * @Subject
     * @Iterations(3)
     * @OutputTimeUnit("milliseconds", precision=3)
     */
    public function apply() : void
    {
        $this->dataset->apply($this->transformer);
    }
}
