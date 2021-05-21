<?php

namespace Rubix\ML\Tests\Persisters;

use League\Flysystem\Filesystem;
use League\Flysystem\Memory\MemoryAdapter;
use Rubix\ML\Encoding;
use Rubix\ML\Persisters\FlysystemV1;
use Rubix\ML\Persisters\Persister;
use PHPUnit\Framework\TestCase;

/**
 * @group Persisters
 * @covers \Rubix\ML\Persisters\FlysystemV1
 */
class FlysystemV1Test extends TestCase
{
    /**
     * @var string
     */
    const PATH = '/path/to/test.model';

    const DATA = 'There are 10 kinds of people in this world. Those who understand binary and those who don’t.';

    /**
     * @var \League\Flysystem\FilesystemInterface
     */
    protected $filesystem;

    /**
     * @var \Rubix\ML\Persistable
     */
    protected $en;

    /**
     * @var \Rubix\ML\Persisters\Flysystem
     */
    protected $persister;

    /**
     * @before
     */
    protected function setUp() : void
    {
        if (!interface_exists('League\Flysystem\FilesystemInterface')) {
            $this->markTestSkipped();
        }

        $this->filesystem = new Filesystem(new MemoryAdapter());

        $this->persister = new FlysystemV1(self::PATH, $this->filesystem);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(FlysystemV1::class, $this->persister);
        $this->assertInstanceOf(Persister::class, $this->persister);
    }

    /**
     * @test
     */
    public function saveLoad() : void
    {
        $encoding = new Encoding(self::DATA);

        $this->persister->save($encoding);

        $encoding = $this->persister->load();

        $this->assertInstanceOf(Encoding::class, $encoding);
        $this->assertSame(self::DATA, $encoding->data());
    }

    protected function assertPreConditions() : void
    {
        $this->assertFalse($this->filesystem->has(self::PATH));
    }
}
