#pragma once
#include <regex>
#include <string>

/** Sequencer
*/
/**
Code Copyright: Andrew Britton
Project: Movie Hasher
2019

Description:
Movie Integrity generator
*/


class Sequencer
{
public:

	//////////////////////
	// CTors / DTor
	Sequencer(std::string const first_frame_in): first_frame_name(first_frame_in) 
	{
		bool result = std::regex_search(first_frame_name, first_frame_regex_results, match_pattern);
		std::string src = first_frame_regex_results[1];
		first_frame = std::stoi(src);
		digit_padding = src.length();
		current_frame = first_frame;
	}

	~Sequencer() {}

	Sequencer(Sequencer const&) = delete;
	void operator=(Sequencer const&) = delete;


	std::string get_next_frame();


private:

	std::string first_frame_name;
	std::regex match_pattern = std::regex("[a-zA-Z]*.([0-9]*).\\w{3}");
	std::smatch first_frame_regex_results;
	int first_frame;
	int digit_padding;
	int current_frame;

};