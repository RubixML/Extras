<?php

namespace Rubix\ML\Tests\Tokenizers;

use Rubix\ML\Tokenizers\Kmer;
use Rubix\ML\Tokenizers\Tokenizer;
use PHPUnit\Framework\TestCase;

/**
 * @group Tokenizers
 * @covers \Rubix\ML\Tokenizers\Kmer
 */
class KmerTest extends TestCase
{
    /**
     * @var \Rubix\ML\Tokenizers\Kmer
     */
    protected $tokenizer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->tokenizer = new Kmer(4);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Kmer::class, $this->tokenizer);
        $this->assertInstanceOf(Tokenizer::class, $this->tokenizer);
    }

    /**
     * @test
     */
    public function tokenize() : void
    {
        $text = 'ACGCGTCGAATTCGNTCGA';

        $expected = [
            'ACGC', 'CGCG', 'GCGT', 'CGTC', 'GTCG', 'TCGA', 'CGAA', 'GAAT', 'AATT', 'ATTC', 'TTCG',
        ];

        $tokens = $this->tokenizer->tokenize($text);

        $this->assertEquals($expected, $tokens);
        $this->assertCount(11, $tokens);
    }
}
