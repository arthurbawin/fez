#include <checkpoint_tools.h>
#include <utilities.h>

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <system_error>
#include <utility>
#include <vector>

namespace
{
  namespace fs = std::filesystem;

  // Number of digits of the time step in the checkpoint directory names
  constexpr unsigned int n_time_step_digits = 6;

  // Extension of the checkpoint directories, shared by the functions writing
  // and recognizing their names
  const std::string checkpoint_extension = ".chk";

  // The output directory, which may be empty for the current directory
  fs::path output_path(const std::string &output_dir)
  {
    return output_dir.empty() ? fs::path(".") : fs::path(output_dir);
  }

  /**
   * Path of the directory <output_dir><prefix>_<N><extension>, with N the time
   * step padded with zeros to n_time_step_digits digits, and the extension
   * ".chk" (checkpoint), ".chk.tmp" (temporary directory) or ".bak<k>.chk"
   * (backup).
   */
  fs::path checkpoint_path(const std::string &output_dir,
                           const std::string &prefix,
                           const unsigned int time_step,
                           const std::string &extension)
  {
    std::ostringstream name;
    name << prefix << "_" << std::setw(n_time_step_digits) << std::setfill('0')
         << time_step << extension;
    return output_path(output_dir) / name.str();
  }

  // The path as a string ending with a single directory separator
  std::string with_trailing_slash(const fs::path &path)
  {
    return (path / "").string();
  }

  /**
   * If @p name is the name of a checkpoint directory <prefix>_<N>.chk, set
   * @p time_step to N and return true. The backups <prefix>_<N>.bak<k>.chk,
   * the temporary directories <prefix>_<N>.chk.tmp and the names whose N is not
   * a valid unsigned int are not matched.
   */
  bool parse_checkpoint_name(const std::string &name,
                             const std::string &prefix,
                             unsigned int      &time_step)
  {
    const std::string  head = prefix + "_";
    const std::string &tail = checkpoint_extension;
    if (name.size() <= head.size() + tail.size() ||
        name.compare(0, head.size(), head) != 0 ||
        name.compare(name.size() - tail.size(), tail.size(), tail) != 0)
      return false;

    // from_chars only accepts digits, and reports values out of range
    const char *first       = name.data() + head.size();
    const char *last        = name.data() + name.size() - tail.size();
    const auto [end, error] = std::from_chars(first, last, time_step);
    return error == std::errc() && end == last;
  }

  // The checkpoint directories <prefix>_<N>.chk, sorted by increasing N
  std::vector<std::pair<unsigned int, fs::path>>
  list_checkpoint_directories(const std::string &output_dir,
                              const std::string &prefix)
  {
    std::vector<std::pair<unsigned int, fs::path>> directories;
    for (const auto &entry : fs::directory_iterator(output_path(output_dir)))
    {
      // Match the name first: is_directory() may query the filesystem, which
      // is slow for output directories with many files on parallel filesystems
      unsigned int time_step;
      if (parse_checkpoint_name(entry.path().filename().string(),
                                prefix,
                                time_step) &&
          entry.is_directory())
        directories.emplace_back(time_step, entry.path());
    }
    std::sort(directories.begin(), directories.end());
    return directories;
  }

  const std::string info_filename   = "info.txt";
  const std::string n_processes_key = "number of MPI processes = ";
} // namespace

namespace CheckpointTools
{
  std::string checkpoint_directory(const std::string &output_dir,
                                   const std::string &prefix,
                                   const unsigned int time_step)
  {
    return with_trailing_slash(
      checkpoint_path(output_dir, prefix, time_step, checkpoint_extension));
  }

  std::string
  create_temporary_checkpoint_directory(const std::string &output_dir,
                                        const std::string &prefix,
                                        const unsigned int time_step,
                                        const MPI_Comm     comm)
  {
    const fs::path directory = checkpoint_path(output_dir,
                                               prefix,
                                               time_step,
                                               checkpoint_extension + ".tmp");
    call_on_root_and_share_errors(
      comm,
      [&]() {
        fs::remove_all(directory);
        fs::create_directories(directory);
      },
      "Creating the checkpoint directory " + directory.string());
    return with_trailing_slash(directory);
  }

  void finalize_checkpoint_directory(const std::string &output_dir,
                                     const std::string &prefix,
                                     const unsigned int time_step,
                                     const unsigned int n_checkpoints_to_keep,
                                     const MPI_Comm     comm)
  {
    // Wait for all processes to be done writing to the temporary directory
    const int ierr = MPI_Barrier(comm);
    AssertThrowMPI(ierr);

    const fs::path directory =
      checkpoint_path(output_dir, prefix, time_step, checkpoint_extension);
    call_on_root_and_share_errors(
      comm,
      [&]() {
        // Move an existing checkpoint of this time step to the first unused
        // backup name
        if (fs::exists(directory))
        {
          fs::path     backup;
          unsigned int k = 0;
          do
          {
            backup = checkpoint_path(output_dir,
                                     prefix,
                                     time_step,
                                     ".bak" + std::to_string(k++) +
                                       checkpoint_extension);
          }
          while (fs::exists(backup));
          fs::rename(directory, backup);
        }

        fs::rename(checkpoint_path(output_dir,
                                   prefix,
                                   time_step,
                                   checkpoint_extension + ".tmp"),
                   directory);

        if (n_checkpoints_to_keep > 0)
        {
          // Sorted by increasing time step. The directories with a later time
          // step come from an abandoned run, and are not removed.
          auto directories = list_checkpoint_directories(output_dir, prefix);
          while (!directories.empty() && directories.back().first > time_step)
            directories.pop_back();

          // The checkpoint was written successfully: a directory that cannot
          // be removed only yields a warning, rather than stopping the
          // simulation
          for (std::size_t i = 0;
               i + n_checkpoints_to_keep < directories.size();
               ++i)
          {
            std::error_code error;
            fs::remove_all(directories[i].second, error);
            if (error)
              std::cout << "Warning: could not remove the checkpoint directory "
                        << directories[i].second.string() << ": "
                        << error.message() << std::endl;
          }
        }
      },
      "Finalizing the checkpoint directory " + directory.string());
  }

  std::string find_restart_directory(const std::string &output_dir,
                                     const std::string &prefix,
                                     const std::string &restart_from,
                                     const MPI_Comm     comm)
  {
    std::string directory;
    call_on_root_and_share_errors(
      comm,
      [&]() {
        if (restart_from == "latest")
        {
          const auto directories =
            list_checkpoint_directories(output_dir, prefix);
          AssertThrow(!directories.empty(),
                      ExcMessage("No checkpoint directory " + prefix +
                                 "_<N>.chk was found in the output directory " +
                                 output_path(output_dir).string() + "."));
          directory = directories.back().second.string();
        }
        else
        {
          const fs::path path(restart_from);
          directory =
            (path.is_absolute() ? path : output_path(output_dir) / path)
              .string();
          AssertThrow(fs::is_directory(directory),
                      ExcMessage("The checkpoint directory " + directory +
                                 " does not exist."));
        }
      },
      "Finding the checkpoint directory to restart from");

    return with_trailing_slash(Utilities::MPI::broadcast(comm, directory, 0));
  }

  void write_checkpoint_info(const std::string &directory,
                             const unsigned int time_step,
                             const double       time,
                             const MPI_Comm     comm)
  {
    call_on_root_and_share_errors(
      comm,
      [&]() {
        std::ofstream file(directory + info_filename);
        file << "time step = " << time_step << std::endl
             << "time = " << std::setprecision(17) << time << std::endl
             << n_processes_key << Utilities::MPI::n_mpi_processes(comm)
             << std::endl;
        AssertThrow(file,
                    ExcMessage("Could not write the file " + directory +
                               info_filename + "."));
      },
      "Writing the checkpoint information");
  }

  void check_checkpoint_info(const std::string &directory, const MPI_Comm comm)
  {
    call_on_root_and_share_errors(
      comm,
      [&]() {
        // info.txt is written last, so its absence means that the checkpoint
        // is incomplete, e.g., a temporary directory .chk.tmp
        std::ifstream file(directory + info_filename);
        AssertThrow(file,
                    ExcMessage("Could not read the file " + directory +
                               info_filename + ": " + directory +
                               " is not a complete checkpoint directory."));

        unsigned int n_processes = 0;
        for (std::string line; std::getline(file, line);)
          if (line.compare(0, n_processes_key.size(), n_processes_key) == 0)
            std::from_chars(line.data() + n_processes_key.size(),
                            line.data() + line.size(),
                            n_processes);

        AssertThrow(n_processes > 0,
                    ExcMessage("The number of MPI processes is missing or "
                               "invalid in " +
                               directory + info_filename + "."));
        AssertThrow(n_processes == Utilities::MPI::n_mpi_processes(comm),
                    ExcMessage(
                      "The checkpoint in " + directory + " was written with " +
                      std::to_string(n_processes) +
                      " MPI processes, but this simulation runs with " +
                      std::to_string(Utilities::MPI::n_mpi_processes(comm)) +
                      ". Restart with the same number of processes."));
      },
      "Checking the checkpoint information");
  }
} // namespace CheckpointTools
