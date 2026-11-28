#include <checkpoint_tools.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "../tests.h"

/**
 * Tests the management of the checkpoint directories in CheckpointTools:
 * backups of a checkpoint written again at the same time step, removal of the
 * oldest checkpoints, and search of the checkpoint to restart from.
 */
int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  const auto                       comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  mpi_initlog();

  namespace fs                 = std::filesystem;
  const std::string output_dir = "./";
  const std::string prefix     = "chk";

  const auto has_prefix = [&](const fs::path &path) {
    return path.filename().string().rfind(prefix + "_", 0) == 0;
  };

  remove_checkpoint_directories(prefix, comm);

  // Write a checkpoint in which each rank writes @p id to its own file
  const auto write_checkpoint = [&](const unsigned int time_step,
                                    const unsigned int n_checkpoints_to_keep,
                                    const unsigned int id) {
    const std::string directory =
      CheckpointTools::create_temporary_checkpoint_directory(output_dir,
                                                             prefix,
                                                             time_step,
                                                             comm);
    {
      std::ofstream file(directory + "rank" + std::to_string(rank));
      file << id;
    }
    CheckpointTools::write_checkpoint_info(directory,
                                           time_step,
                                           0.1 * time_step,
                                           comm);
    CheckpointTools::finalize_checkpoint_directory(
      output_dir, prefix, time_step, n_checkpoints_to_keep, comm);
  };

  const auto read_id = [&](const std::string &directory) {
    unsigned int  id;
    std::ifstream file(directory + "rank" + std::to_string(rank));
    file >> id;
    return id;
  };

  // Log the checkpoint directories, sorted by name
  const auto log_directories = [&](const std::string &title) {
    std::vector<std::string> names;
    for (const auto &entry : fs::directory_iterator(output_dir))
      if (entry.is_directory() && has_prefix(entry.path()))
        names.push_back(entry.path().filename().string());
    std::sort(names.begin(), names.end());

    if (rank == 0)
    {
      deallog << title << std::endl;
      for (const auto &name : names)
        deallog << "  " << name << std::endl;
    }
  };

  // Writing the checkpoint of the same time step three times keeps the first
  // two as backups, the oldest one being bak0
  for (unsigned int id = 0; id < 3; ++id)
    write_checkpoint(5, 0, id);
  log_directories("After writing time step 5 three times:");
  AssertThrow(read_id("chk_000005.bak0.chk/") == 0 &&
                read_id("chk_000005.bak1.chk/") == 1 &&
                read_id("chk_000005.chk/") == 2,
              ExcMessage("Wrong order of the checkpoint backups"));

  // Keep the two most recent checkpoints: writing time step 15 removes
  // time step 5, but not its backups
  write_checkpoint(10, 2, 0);
  write_checkpoint(15, 2, 0);
  log_directories("After writing time steps 10 and 15, keeping 2:");

  // Time step 30 plays the checkpoint of an abandoned run: it is kept when
  // writing time step 20 while keeping a single checkpoint
  write_checkpoint(30, 0, 0);
  write_checkpoint(20, 1, 0);
  log_directories("After writing time steps 30 and 20, keeping 1:");

  // Find the checkpoint to restart from
  const std::string latest =
    CheckpointTools::find_restart_directory(output_dir, prefix, "latest", comm);
  const std::string explicit_directory =
    CheckpointTools::find_restart_directory(output_dir,
                                            prefix,
                                            "chk_000020.chk",
                                            comm);
  CheckpointTools::check_checkpoint_info(explicit_directory, comm);
  if (rank == 0)
    deallog << "Latest checkpoint: " << latest << std::endl
            << "Explicit checkpoint: " << explicit_directory << std::endl;

  // A missing checkpoint directory throws on all ranks
  bool missing_detected = false;
  try
  {
    CheckpointTools::find_restart_directory(output_dir,
                                            prefix,
                                            "chk_000999.chk",
                                            comm);
  }
  catch (const std::exception &)
  {
    missing_detected = true;
  }
  AssertThrow(missing_detected,
              ExcMessage("A missing checkpoint was not detected"));
  if (rank == 0)
    deallog << "Missing checkpoint detected" << std::endl;

  // Reading a checkpoint written with 2 ranks on a single rank throws
  bool n_ranks_detected = false;
  try
  {
    CheckpointTools::check_checkpoint_info(explicit_directory, MPI_COMM_SELF);
  }
  catch (const std::exception &)
  {
    n_ranks_detected = true;
  }
  AssertThrow(n_ranks_detected,
              ExcMessage("A different number of ranks was not detected"));
  if (rank == 0)
    deallog << "Different number of ranks detected" << std::endl;
}
