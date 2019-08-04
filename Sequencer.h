#pragma once
#include <regex>
#include <string>
#include <experimental/filesystem>


/** Sequencer
*/
/**
Code Copyright: Andrew Britton
Project: Movie Hasher
2019

Description:
Movie Integrity generator

Code Review Notes:
https://codereview.stackexchange.com/questions/224654/filename-sequencer-with-outdated-c
*/


class Sequencer
{
public:

	//////////////////////
	// CTors / DTor
	Sequencer(std::string const first_frame_in)
	{
		std::regex match_pattern = std::regex("([a-zA-Z]+).([0-9]+).(\\w{3})");
		std::smatch first_frame_regex_results;
		int first_frame;

		bool result = std::regex_search(first_frame_in, first_frame_regex_results, match_pattern);
		m_prefix = first_frame_regex_results[1];
		std::string frame = first_frame_regex_results[2];
		m_ext = first_frame_regex_results[3];
		first_frame = std::stoi(frame);
		m_digit_padding = frame.length();
		m_current_frame = first_frame;
		m_out_prefix = "_signature";
		m_file_delim = ".";

		get_last_frame();
	}

	~Sequencer() {}

	Sequencer(Sequencer const&) = delete;
	void operator=(Sequencer const&) = delete;


	bool get_next_frame(std::string &next_frame, std::string &name_out)
	{
		if (m_current_frame + 1 <= m_last_frame)
		{
			std::string frame_number = add_padding(m_current_frame);
			next_frame = m_prefix + m_file_delim + frame_number + m_file_delim + m_ext;
			name_out = m_prefix + m_out_prefix + m_file_delim + frame_number + m_file_delim + "ppm";
			m_current_frame++;
			return true;
		}
		else
		{
			name_out.clear();
			next_frame.clear();
			return false;
		}
	}

	bool get_next_frame(std::string &next_frame, std::string &plus_one_frame, std::string &name_out)
	{
		if (m_current_frame+1 <= m_last_frame)
		{
			std::string frame_number = add_padding(m_current_frame);
			std::string plus_one_number = add_padding(m_current_frame+1);
			next_frame = m_prefix + m_file_delim + frame_number + m_file_delim + m_ext;
			plus_one_frame = m_prefix + m_file_delim + plus_one_number + m_file_delim + m_ext;
			name_out = m_prefix + m_out_prefix + m_file_delim + frame_number + m_file_delim + "ppm";
			m_current_frame++;
			return true;
		}
		else
		{
			name_out.clear();
			next_frame.clear();
			return false;
		}
	}


private:

	void get_last_frame()
	{
		bool test = true;
		std::string outname;
		unsigned int i = m_current_frame + 1;
		do
		{
			outname = "C:\\ProgramData\\NVIDIA Corporation\\CUDA Samples\\v9.0\\3_Imaging\\MovieHasher\\data\\" + m_prefix + m_file_delim + add_padding(i) + m_file_delim + m_ext;
			test = std::experimental::filesystem::exists(outname);
			if (test)
				i++;
			else
				i--;
		} while (test);
		m_last_frame = i;
	}

	std::string add_padding(int frame_num_in)
	{
		std::string outval = std::to_string(frame_num_in);

		int pad_length = std::max<int>(0, m_digit_padding - outval.length());

		return std::string(pad_length, '0') + outval;
	}

	
	std::string m_prefix, m_ext, m_out_prefix, m_file_delim;
	int m_last_frame;
	int m_digit_padding;
	int m_current_frame;

};