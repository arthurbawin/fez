
#include <mpi.h>
#include <utilities.h>

#include <filesystem>

void replace_temporary_files(const std::string directory,
                             const std::string temporary_filename_prefix,
                             const std::string final_filename_prefix,
                             const MPI_Comm   &mpi_communicator)
{
  // Create a shared-memory communicator (one per node)
  MPI_Comm shm_comm;
  MPI_Comm_split_type(
    mpi_communicator, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm_comm);

  int shm_rank;
  MPI_Comm_rank(shm_comm, &shm_rank);

  // Only one rank per shared-memory node touches the filesystem
  if (shm_rank == 0)
  {
    std::list<std::string> tmp_checkpoint_files;
    for (const auto &dir_entry : std::filesystem::directory_iterator(directory))
      if (dir_entry.is_regular_file() &&
          (dir_entry.path().filename().string().find(
             temporary_filename_prefix) == 0))
        tmp_checkpoint_files.push_back(dir_entry.path().filename().string());

    for (const std::string &filename : tmp_checkpoint_files)
    {
      const std::string final_filename =
        directory + final_filename_prefix +
        filename.substr(temporary_filename_prefix.size(), std::string::npos);
      std::filesystem::rename(directory + filename, final_filename);
    }
  }

  // Ensure all ranks see the renamed files
  MPI_Barrier(mpi_communicator);
  MPI_Comm_free(&shm_comm);
}