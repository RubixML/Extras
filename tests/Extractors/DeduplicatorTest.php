<?php

namespace Rubix\ML\Tests\Extractors;

use Rubix\ML\Extractors\CSV;
use Rubix\ML\Extractors\Extractor;
use Rubix\ML\Extractors\Deduplicator;
use PHPUnit\Framework\TestCase;
use IteratorAggregate;
use Traversable;

/**
 * @group Extractors
 * @covers \Rubix\ML\Extractors\Deduplicator
 */
class DeduplicatorTest extends TestCase
{
    /**
     * @var \Rubix\ML\Extractors\Deduplicator;
     */
    protected $extractor;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->extractor = new Deduplicator(new CSV('tests/test.csv', true));
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Deduplicator::class, $this->extractor);
        $this->assertInstanceOf(Extractor::class, $this->extractor);
        $this->assertInstanceOf(IteratorAggregate::class, $this->extractor);
        $this->assertInstanceOf(Traversable::class, $this->extractor);
    }

    /**
     * @test
     */
    public function extract() : void
    {
        $expected = [
            ['attitude' => 'nice', 'texture' => 'furry', 'sociability' => 'friendly', 'rating' => '4', 'class' => 'not monster'],
            ['attitude' => 'mean', 'texture' => 'furry', 'sociability' => 'loner', 'rating' => '-1.5', 'class' => 'monster'],
            ['attitude' => 'nice', 'texture' => 'rough', 'sociability' => 'friendly', 'rating' => '2.6', 'class' => 'not monster'],
            ['attitude' => 'mean', 'texture' => 'rough', 'sociability' => 'friendly', 'rating' => '-1', 'class' => 'monster'],
            ['attitude' => 'nice', 'texture' => 'rough', 'sociability' => 'friendly', 'rating' => '2.9', 'class' => 'not monster'],
            ['attitude' => 'nice', 'texture' => 'furry', 'sociability' => 'loner', 'rating' => '-5', 'class' => 'not monster'],
        ];

        $this->assertEquals(0, $this->extractor->dropped());

        $records = iterator_to_array($this->extractor, false);

        $this->assertEquals($expected, $records);

        $this->assertEquals(1, $this->extractor->dropped());
    }
}
