#include <iostream>
#include <chrono>

class Profiler
{
private:
    std::chrono::time_point<std::chrono::steady_clock> m_start;
    std::chrono::time_point<std::chrono::steady_clock> m_end;
public:
    void start()
    {
        m_start = std::chrono::steady_clock::now();
    };
    void end()
    {
        m_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = m_end - m_start;
        std::cout << "elapsed: " << elapsed_seconds.count()<< " s"<< std::endl;
    };
    const auto getTime() const
    {
        return std::chrono::steady_clock::now();
    };
};

//用法： 
// Profiler p;
// p.start();
// //do something
// p.end();