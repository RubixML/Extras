<?php

namespace Rubix\ML\Tests\Kernels\Distance;

use Rubix\ML\Kernels\Distance\Levenshtein;
use Rubix\ML\Kernels\Distance\Distance;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Distances
 * @covers \Rubix\ML\Kernels\Distance\Levenshtein
 */
class LevenshteinTest extends TestCase
{
    /**
     * @var \Rubix\ML\Kernels\Distance\Levenshtein
     */
    protected $kernel;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->kernel = new Levenshtein();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Levenshtein::class, $this->kernel);
        $this->assertInstanceOf(Distance::class, $this->kernel);
    }

    /**
     * @test
     * @dataProvider computeProvider
     *
     * @param list<string> $a
     * @param list<string> $b
     * @param float $expected
     */
    public function compute(array $a, array $b, $expected) : void
    {
        $distance = $this->kernel->compute($a, $b);

        $this->assertGreaterThanOrEqual(0.0, $distance);
        $this->assertEquals($expected, $distance);
    }

    /**
     * @return \Generator<array<mixed>>
     */
    public function computeProvider() : Generator
    {
        yield [['aaa'], ['aaaaaa'], 3.0];

        yield [['toast', 'naan'], ['pretzels', 'pizza'], 12.0];

        yield [['Beef'], ['feeB'], 2.0];

        yield [['Levenshtein'], ['Levanshtein'], 1.0];
    }
}
